import os
import config
import numpy as np
from tensorflow.keras.optimizers import Adam
from .base import BaseAgent
from sacd.model import TwinnedQNetwork, CateoricalPolicy
from sacd.utils import disable_gradients
import time
import tensorflow as tf
from sacd.utils import update_params


class SacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed)

        with self.device:
            # Define networks.
            self.policy = CateoricalPolicy(config.POTENTIAL_MOVE_NUM)
            self.online_critic = TwinnedQNetwork(
                config.POTENTIAL_MOVE_NUM, dueling_net=dueling_net)
            self.target_critic = TwinnedQNetwork(
                config.POTENTIAL_MOVE_NUM, dueling_net=dueling_net)
            self.policy.build(input_shape=config.NN_INPUT_SHAPE)
            self.online_critic.build(input_shape=config.NN_INPUT_SHAPE)
            self.target_critic.build(input_shape=config.NN_INPUT_SHAPE)

        # Copy parameters of the learning network to the target network.
        self.target_critic.set_weights(self.online_critic.get_weights())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)
        self.optim = Adam(learning_rate=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / config.POTENTIAL_MOVE_NUM) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = tf.Variable(0.)
        self.alpha = tf.exp(self.log_alpha)
        # self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def learn(self):

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample()
        else:
            batch, _ = self.memory.sample()
            # Set priority weights to 1 when we don't use PER.
            weights = 1.
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.online_critic.Q1.trainable_weights)
            tape.watch(self.online_critic.Q2.trainable_weights)
            tape.watch(self.policy.trainable_weights)
            tape.watch(self.log_alpha)
            q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
                self.calc_critic_loss(batch, weights)
            policy_loss, entropies = self.calc_policy_loss(batch, weights)
            entropy_loss = self.calc_entropy_loss(entropies, weights)
        update_params(self.optim, q1_loss,
                      self.online_critic.Q1.trainable_weights, tape)
        update_params(self.optim, q2_loss,
                      self.online_critic.Q2.trainable_weights, tape)
        update_params(self.optim, policy_loss,
                      self.policy.trainable_weights, tape)
        update_params(self.optim, entropy_loss, self.log_alpha, tape)

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            with self.writer.as_default(self.learning_steps):
                tf.summary.scalar('loss/Q1', q1_loss)
                tf.summary.scalar('loss/Q2', q2_loss)
                tf.summary.scalar('loss/policy', policy_loss)
                tf.summary.scalar('loss/alpha', entropy_loss)
                tf.summary.scalar('stats/alpha', self.alpha)
                tf.summary.scalar('stats/mean_Q1', mean_q1)
                tf.summary.scalar('stats/mean_Q2', mean_q2)
                tf.summary.scalar('stats/entropy', tf.reduce_mean(entropies))
                self.writer.flush()

    def explore(self, state):
        # Act with randomness.
        state = tf.expand_dims(state, axis=0)
        with tf.stop_gradient():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = tf.expand_dims(state, axis=0)
        with tf.stop_gradient():
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        self.target_critic.set_weights(self.online_critic.get_weights())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = tf.gather(curr_q1, actions, axis=1, batch_dims=1)
        curr_q2 = tf.gather(curr_q2, actions, axis=1, batch_dims=1)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        _, action_probs, log_action_probs = self.policy.sample(next_states)
        next_q1, next_q2 = self.target_critic(next_states)
        next_q = tf.math.reduce_sum(action_probs * (
            tf.math.minimum(next_q1, next_q2) -
            self.alpha * log_action_probs
        ), axis=1)
        assert rewards.shape == next_q.shape
        return rewards + (1.0 - tf.cast(dones, tf.float32)) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = tf.abs(curr_q1 - target_q)

        # We log means of Q to monitor training.
        mean_q1 = tf.reduce_mean(curr_q1)
        mean_q2 = tf.reduce_mean(curr_q2)

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = tf.reduce_mean(tf.pow(curr_q1 - target_q, 2) * weights)
        q2_loss = tf.reduce_mean(tf.pow(curr_q2 - target_q, 2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        # Q for every actions to calculate expectations of Q.
        q1, q2 = self.online_critic(states)
        q = tf.math.minimum(q1, q2)

        # Expectations of entropies.
        entropies = -tf.math.reduce_sum(
            action_probs * log_action_probs, axis=1)

        # Expectations of Q.
        q = tf.math.reduce_sum(tf.math.minimum(
            q1, q2) * action_probs, axis=1, keepdims=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = tf.reduce_mean(weights * (- q - self.alpha * entropies))

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies, weights):

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -tf.reduce_mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))

    def load_models(self, p):
        self.policy.load(os.path.join(p, 'policy.pth'))
        self.online_critic.load(os.path.join(p, 'online_critic.pth'))
        self.target_critic.load(os.path.join(p, 'target_critic.pth'))
        print(f"load weight file in {p}")
        last_change_time = time.strftime(
            "%Y/%m/%d %H:%M:%S", time.localtime(os.stat(os.path.join(p, 'policy.pth')).st_mtime))
        print(f"file last changed at {last_change_time}")
