# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""A Soft Actor-Critic Agent.

Implements the Soft Actor-Critic (SAC) algorithm from
"Soft Actor-Critic Algorithms and Applications" by Haarnoja et al (2019).
"""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import collections
import os
from time import time
from typing import Callable, Optional, Text

import gin
import numpy as np
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from policy import SamplePolicy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity
from tf_agents.policies.greedy_policy import GreedyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
import tensorflow_probability as tfp
import train_config
SacLossInfo = collections.namedtuple(
    'SacLossInfo', ('critic_loss', 'actor_loss', 'alpha_loss','error'))


class Checkpoint:
    def __init__(self, dir: str, save_interval,
                 _critic_network_1,
                 _critic_network_2,
                 _target_critic_network_1,
                 _target_critic_network_2,
                 _actor_network,
                 _log_alpha,
                 step_counter
                 ) -> None:
        self._path = dir
        self.train_step_counter = step_counter
        self._save_interval = save_interval
        self._checkpoint = tf.train.Checkpoint(
            _critic_network_1=_critic_network_1,
            _critic_network_2=_critic_network_2,
            _target_critic_network_1=_target_critic_network_1,
            _target_critic_network_2=_target_critic_network_2,
            _actor_network=_actor_network,
            _log_alpha=_log_alpha,
            step_counter=step_counter,
        )
        self._latest_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(dir, 'latest'), max_to_keep=5)
        self._best_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(dir, 'best'), max_to_keep=5)

    def load(self, path, catergory: str = 'latest'):

        _latest_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(path, 'latest'), max_to_keep=5)
        _best_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(path, 'best'), max_to_keep=5)
        if catergory == "latest":
            target = _latest_manager
        else:
            target = _best_manager
        status = self._checkpoint.restore(target.latest_checkpoint)
        status.assert_consumed()
        print(f'Using checkpointer from {target.latest_checkpoint}')

    def save_best(self):
        self._best_manager.save()
        pass

    def save_latest(self):
        if self.train_step_counter % self._save_interval == 0:
            self._latest_manager.save()
        pass


@gin.configurable
class SacdAgent(tf_agent.TFAgent):
    """A SAC Agent."""

    def __init__(self,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 critic_network: network.Network,
                 actor_network: network.Network,
                 actor_optimizer: types.Optimizer,
                 critic_optimizer1: types.Optimizer,
                 critic_optimizer2: types.Optimizer,
                 alpha_optimizer: types.Optimizer,
                 save_dir: str,
                 actor_loss_weight: types.Float = 1.0,
                 critic_loss_weight: types.Float = 1.0,
                 alpha_loss_weight: types.Float = 1.0,
                 actor_policy_ctor: Callable[
                     ..., tf_policy.TFPolicy] = SamplePolicy,
                 critic_network_2: Optional[network.Network] = None,
                 target_critic_network: Optional[network.Network] = None,
                 target_critic_network_2: Optional[network.Network] = None,
                 target_update_tau: types.Float = 1.0,
                 target_update_period: types.Int = train_config.target_update_interval,
                 td_errors_loss_fn: types.LossFn = tf.math.squared_difference,
                 gamma: types.Float = train_config.gamma,
                 reward_scale_factor: types.Float = 1.0,
                 initial_log_alpha: types.Float = 0.0,
                 log_interval: types.Int = train_config.log_interval,
                 eval_interval: types.Int = train_config.eval_interval,
                 save_interval: types.Int = train_config.save_interval,
                 eval_num_episode: types.Int = train_config.num_eval_steps,
                 target_entropy_ratio: Optional[types.Float] = train_config.target_entropy_ratio,
                 gradient_clipping: Optional[types.Float] = None,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 train_step_counter: Optional[tf.Variable] = None,
                 name: Optional[Text] = None):

        tf.Module.__init__(self, name=name)

        net_observation_spec = time_step_spec.observation
        critic_spec = (net_observation_spec)

        self._critic_network_1 = critic_network

        if critic_network_2 is not None:
            self._critic_network_2 = critic_network_2
        else:
            self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
            # Do not use target_critic_network_2 if critic_network_2 is None.
            target_critic_network_2 = None

        # Wait until critic_network_2 has been copied from critic_network_1 before
        # creating variables on both.
        self._critic_network_1.create_variables(critic_spec)
        self._critic_network_2.create_variables(critic_spec)

        if target_critic_network:
            target_critic_network.create_variables(critic_spec)

        self._target_critic_network_1 = (
            common.maybe_copy_target_network_with_checks(
                self._critic_network_1,
                target_critic_network,
                input_spec=critic_spec,
                name='TargetCriticNetwork1'))

        if target_critic_network_2:
            target_critic_network_2.create_variables(critic_spec)
        self._target_critic_network_2 = (
            common.maybe_copy_target_network_with_checks(
                self._critic_network_2,
                target_critic_network_2,
                input_spec=critic_spec,
                name='TargetCriticNetwork2'))

        if actor_network:
            actor_network.create_variables(net_observation_spec)
        self._actor_network = actor_network
        self._greedy_policy = GreedyPolicy(actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False)

        )

        self._random_policy = RandomTFPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

        sample_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False)

        self._train_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=True)

        self._log_alpha = common.create_variable(
            'initial_log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)

        target_entropy = self._get_default_target_entropy(
            action_spec)*target_entropy_ratio

        self._save_dir = save_dir
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer1 = critic_optimizer1
        self._critic_optimizer2 = critic_optimizer2
        self._alpha_optimizer = alpha_optimizer
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._alpha_loss_weight = alpha_loss_weight
        self._td_errors_loss_fn = td_errors_loss_fn
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._target_entropy = target_entropy
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._update_target = self._get_target_updater(
            tau=self._target_update_tau, period=self._target_update_period)
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._save_interval = save_interval
        self._eval_num_episode = eval_num_episode
        train_sequence_length = 2 if not critic_network.state_spec else None

        super(SacdAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=sample_policy,
            collect_policy=sample_policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
        )

        self.__ckp = Checkpoint(
            self._save_dir, self._save_interval,
            _critic_network_1=self._critic_network_1,
            _critic_network_2=self._critic_network_2,
            _target_critic_network_1=self._target_critic_network_1,
            _target_critic_network_2=self._target_critic_network_2,
            _actor_network=self._actor_network,
            _log_alpha=self._log_alpha,
            step_counter=self.train_step_counter,
        )
        self.load = self.__ckp.load
        self.save_best = self.__ckp.save_best
        self.save_latest = self.__ckp.save_latest

        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=(train_sequence_length == 2))

    def _get_default_target_entropy(self, action_spec):
        target_entropy = - \
            np.log(1.0 / (action_spec.maximum-action_spec.minimum+1))
        return target_entropy

    def _initialize(self):
        """Returns an op to initialize the agent.

        Copies weights from the Q networks to the target Q network.
        """
        common.soft_variables_update(
            self._critic_network_1.variables,
            self._target_critic_network_1.variables,
            tau=1.0)
        common.soft_variables_update(
            self._critic_network_2.variables,
            self._target_critic_network_2.variables,
            tau=1.0)

    def _train(self, experience, weights):
        """Returns a train op to update the agent's networks.

        This method trains with the provided batched experience.

        Args:
          experience: A time-stacked trajectory object.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.

        Returns:
          A train_op.

        Raises:
          ValueError: If optimizers are None and no default value was provided to
            the constructor.
        """
        self.inspect_input_data(experience)
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        batch_size = nest_utils.get_outer_shape(
            time_steps, self._time_step_spec)[0]
        if weights is None:
            weights = tf.ones(shape=(batch_size,))

        trainable_critic1_variables = self._critic_network_1.trainable_variables
        trainable_critic2_variables = self._critic_network_2.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(trainable_critic1_variables)
            tape.watch(trainable_critic2_variables)
            critic_loss1, critic_loss2, error, q1, q2 = self.critic_loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)
        # critic_loss1*=self._critic_loss_weight
        # critic_loss2*=self._critic_loss_weight

        tf.debugging.check_numerics(
            critic_loss1, 'Critic1 loss is inf or nan.')
        tf.debugging.check_numerics(
            critic_loss2, 'Critic2 loss is inf or nan.')
        critic1_grads = tape.gradient(
            critic_loss1, trainable_critic1_variables)
        critic2_grads = tape.gradient(
            critic_loss2, trainable_critic2_variables)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss, entropies = self.actor_loss(
                time_steps, weights=weights, training=True)
            actor_loss *= self._actor_loss_weight
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)

        alpha_variable = [self._log_alpha]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert alpha_variable, 'No alpha variable to optimize.'
            tape.watch(alpha_variable)
            alpha_loss = self.alpha_loss(entropies=entropies, weights=weights)
            alpha_loss *= self._alpha_loss_weight
        tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
        alpha_grads = tape.gradient(alpha_loss, alpha_variable)

        self._apply_gradients(critic1_grads, trainable_critic1_variables,self._critic_optimizer1)
        self._apply_gradients(critic2_grads, trainable_critic2_variables,self._critic_optimizer2)
        self._apply_gradients(actor_grads, trainable_actor_variables,self._actor_optimizer)
        self._apply_gradients(alpha_grads, alpha_variable,self._alpha_optimizer)

        self._print_log(critic_loss1, critic_loss2,
                        actor_loss, alpha_loss, entropies,q1,q2)

        self.train_step_counter.assign_add(1)
        self._update_target()
        self.save_latest()
        critic_loss = tf.reduce_mean([critic_loss1, critic_loss2])

        total_loss = critic_loss + actor_loss + alpha_loss

        extra = SacLossInfo(
            critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss,error=error)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _apply_gradients(self, gradients, variables, optimizer):
        # list(...) is required for Python3.
        grads_and_vars = list(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws

        Args:
          tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
          period: Step interval at which the target network is updated.

        Returns:
          A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_target'):

            def update():
                """Update target network."""
                critic_update_1 = common.soft_variables_update(
                    self._critic_network_1.variables,
                    self._target_critic_network_1.variables,
                    tau,
                    tau_non_trainable=1.0)

                critic_2_update_vars = common.deduped_network_variables(
                    self._critic_network_2, self._critic_network_1)

                target_critic_2_update_vars = common.deduped_network_variables(
                    self._target_critic_network_2, self._target_critic_network_1)

                critic_update_2 = common.soft_variables_update(
                    critic_2_update_vars,
                    target_critic_2_update_vars,
                    tau,
                    tau_non_trainable=1.0)

                return tf.group(critic_update_1, critic_update_2)

            return common.Periodically(update, period, 'update_targets')

    def _actions_and_log_probs(self, time_steps, training=False):
        """Get actions and corresponding log probabilities from policy."""
        # Get raw action distribution from policy, and initialize bijectors list.
        states = time_steps.observation
        actions, action_probs, log_action_probs = self._actor_network.sample(
            states, training)

        return actions, action_probs, log_action_probs

    def critic_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    td_errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the critic loss for SAC training.

        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          next_time_steps: A batch of next timesteps.
          td_errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
          gamma: Discount for future rewards.
          reward_scale_factor: Multiplicative factor to scale rewards.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used for training.

        Returns:
          critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

            # Calc current Q
            pred_input = (time_steps.observation)
            pred_td_targets1, _ = self._critic_network_1(
                pred_input, step_type=time_steps.step_type, training=training, action=actions)
            pred_td_targets2, _ = self._critic_network_2(
                pred_input, step_type=time_steps.step_type, training=training, action=actions)
            # td_error = tf.abs(pred_td_targets1-td_targets)

            # Calc target Q
            # We do not update actor or target networks in critic loss.
            next_actions, action_probs, next_log_pis = self._actions_and_log_probs(next_time_steps,
                                                                                   training=False)
            target_input = (next_time_steps.observation)
            next_q1, unused_network_state1 = self._target_critic_network_1(
                target_input, step_type=next_time_steps.step_type, training=False)
            next_q2, unused_network_state2 = self._target_critic_network_2(
                target_input, step_type=next_time_steps.step_type, training=False)
            next_q = tf.stop_gradient(
                tf.math.reduce_sum(
                    action_probs * (
                        tf.minimum(next_q1, next_q2) -
                        tf.exp(self._log_alpha) * next_log_pis), axis=1))

            # Seperate finished states from unfinished ones
            target_q = (
                reward_scale_factor * next_time_steps.reward +
                (1-tf.cast(next_time_steps.is_last(), tf.float32)) * gamma * next_q)
            critic_loss1 = td_errors_loss_fn(target_q, pred_td_targets1)
            critic_loss2 = td_errors_loss_fn(target_q, pred_td_targets2)
            # critic_loss = critic_loss1 + critic_loss2
            errors = tf.abs(pred_td_targets1 - target_q)
            # Sum over the time dimension.
            critic_loss1 = tf.reduce_mean(critic_loss1*weights)
            critic_loss2 = tf.reduce_mean(critic_loss2*weights)

            # agg_loss = common.aggregate_losses(
            #     per_example_loss=critic_loss,
            #     sample_weight=weights,
            #     regularization_loss=(self._critic_network_1.losses +
            #                          self._critic_network_2.losses))
            # critic_loss = agg_loss.total_loss

            self._critic_loss_debug_summaries(target_q, pred_td_targets1,
                                              pred_td_targets2)

            return critic_loss1, critic_loss2, errors, pred_td_targets1,pred_td_targets2

    def actor_loss(self,
                   time_steps: ts.TimeStep,
                   weights: Optional[types.Tensor] = None,
                   training: Optional[bool] = True) -> types.Tensor:
        """Computes the actor_loss for SAC training.

        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          training: Whether training should be applied.

        Returns:
          actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            actions, action_probs, log_pi = self._actions_and_log_probs(time_steps,
                                                                        training=training)

            target_input = (time_steps.observation)
            # We do not update critic during actor loss.
            q1, _ = self._critic_network_1(
                target_input, step_type=time_steps.step_type, training=False)
            q2, _ = self._critic_network_2(
                target_input, step_type=time_steps.step_type, training=False)
            q = tf.stop_gradient(tf.minimum(q1, q2))

            entropies = -tf.math.reduce_sum(
                action_probs * log_pi, axis=1, keepdims=True)

            q = tf.math.reduce_sum(tf.minimum(
                q1, q2)*action_probs, axis=1, keepdims=True)

            actor_loss = tf.math.reduce_mean(
                weights*(- q-tf.exp(self._log_alpha) * entropies))

            self._actor_loss_debug_summaries(actor_loss, actions, log_pi,
                                             q, time_steps)

            return actor_loss, entropies

    def alpha_loss(self,
                   entropies: types.Tensor,
                   weights: Optional[types.Tensor] = None
                   ) -> types.Tensor:
        """Computes the alpha_loss for EC-SAC training.

        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used during training.

        Returns:
          alpha_loss: A scalar alpha loss.
        """
        with tf.name_scope('alpha_loss'):

            entropy_loss = -tf.math.reduce_mean(
                self._log_alpha * (self._target_entropy - entropies)
                * weights)

            self._alpha_loss_debug_summaries(entropy_loss)

            return entropy_loss

    def _critic_loss_debug_summaries(self, td_targets, pred_td_targets1,
                                     pred_td_targets2):
        if self._debug_summaries:
            td_errors1 = td_targets - pred_td_targets1
            td_errors2 = td_targets - pred_td_targets2
            td_errors = tf.concat([td_errors1, td_errors2], axis=0)
            common.generate_tensor_summaries('td_errors', td_errors,
                                             self.train_step_counter)
            common.generate_tensor_summaries('td_targets', td_targets,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                             self.train_step_counter)

    def _actor_loss_debug_summaries(self, actor_loss, actions, log_pi,
                                    target_q_values, time_steps):
        if self._debug_summaries:
            common.generate_tensor_summaries('actor_loss', actor_loss,
                                             self.train_step_counter)
            try:
                for name, action in nest_utils.flatten_with_joined_paths(actions):
                    common.generate_tensor_summaries(name, action,
                                                     self.train_step_counter)
            except ValueError:
                pass  # Guard against internal SAC variants that do not directly
                # generate actions.

            common.generate_tensor_summaries('log_pi', log_pi,
                                             self.train_step_counter)
            tf.summary.scalar(
                name='entropy_avg',
                data=-tf.reduce_mean(input_tensor=log_pi),
                step=self.train_step_counter)
            common.generate_tensor_summaries('target_q_values', target_q_values,
                                             self.train_step_counter)
            batch_size = nest_utils.get_outer_shape(time_steps,
                                                    self._time_step_spec)[0]
            policy_state = self._train_policy.get_initial_state(batch_size)
            action_distribution = self._train_policy.distribution(
                time_steps, policy_state).action
            if isinstance(action_distribution, tfp.distributions.Normal):
                common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('act_stddev',
                                                 action_distribution.scale,
                                                 self.train_step_counter)
            elif isinstance(action_distribution, tfp.distributions.Categorical):
                common.generate_tensor_summaries('act_mode', action_distribution.mode(),
                                                 self.train_step_counter)
            try:
                for name, action_dist in nest_utils.flatten_with_joined_paths(
                        action_distribution):
                    common.generate_tensor_summaries('entropy_' + name,
                                                     action_dist.entropy(),
                                                     self.train_step_counter)
            except NotImplementedError:
                pass  # Some distributions do not have an analytic entropy.

    def _alpha_loss_debug_summaries(self, alpha_loss):
        if self._debug_summaries:
            common.generate_tensor_summaries('alpha_loss', alpha_loss,
                                             self.train_step_counter)

            tf.summary.scalar(
                name='log_alpha', data=self._log_alpha, step=self.train_step_counter)

    def _print_log(self, critic1_loss, critic2_loss, actor_loss, alpha_loss, entropies,q1,q2):
        if self.train_step_counter % self._log_interval == 0:
            with tf.name_scope('Losses'):
                tf.summary.scalar(
                    name='critic1_loss', data=critic1_loss, step=self.train_step_counter)
                tf.summary.scalar(
                    name='critic2_loss', data=critic2_loss, step=self.train_step_counter)
                tf.summary.scalar(
                    name='actor_loss', data=actor_loss, step=self.train_step_counter)
                tf.summary.scalar(
                    name='alpha_loss', data=alpha_loss, step=self.train_step_counter)
            with tf.name_scope('stat'):
                tf.summary.scalar(
                    name='entropy', data=tf.reduce_mean(entropies), step=self.train_step_counter)
                tf.summary.scalar(
                    name='target_entropy', data=self._target_entropy, step=self.train_step_counter)
                tf.summary.scalar(
                    name='alpha', data=tf.exp(self._log_alpha), step=self.train_step_counter)
                tf.summary.scalar(
                    name='Q1', data=tf.reduce_mean(q1), step=self.train_step_counter)
                tf.summary.scalar(
                    name='Q2', data=tf.reduce_mean(q2), step=self.train_step_counter)

    def eval(self, eval_env):
        if self._train_step_counter % self._eval_interval == 0:
            average_episode_length = tf_metrics.AverageEpisodeLengthMetric()
            average_return = tf_metrics.AverageReturnMetric()
            number_episodes = tf_metrics.NumberOfEpisodes()
            observers = [average_episode_length,
                         average_return, number_episodes]
            metric_utils.compute(observers, eval_env,
                                 self._greedy_policy, self._eval_num_episode)

            average_episode_length = average_episode_length.result()
            average_return = average_return.result()
            number_episodes = number_episodes.result()

            with tf.name_scope('Eval'):
                tf.summary.scalar(
                    name='average_episode_length', data=average_episode_length, step=self.train_step_counter)
                tf.summary.scalar(
                    name='average_return', data=average_return, step=self.train_step_counter)
                tf.summary.scalar(
                    name='number_episodes', data=number_episodes, step=self.train_step_counter)

            print(f'average_episode_length:{average_episode_length}')
            print(f'average_return:{average_return}')
            print(f'number_episodes:{number_episodes}')
            if hasattr(self, '_best_average_return'):
                if average_return >= self._best_average_return:
                    self._best_average_return = average_return
                    self.save_best()
            else:
                self._best_average_return = average_return
                self.save_best()

    def inspect_input_data(self, data):
        if self._train_step_counter % self._log_interval == 0:
            actions = data.action
            rewards = data.reward
            variance = tfp.stats.variance(
                tf.reshape(data.action, (-1)), sample_axis=0)
            with tf.name_scope('Input_data'):
                common.generate_tensor_summaries('action', tf.cast(actions, dtype=tf.float32),
                                                 self.train_step_counter)
                tf.summary.scalar(
                    name='action_variance', data=variance, step=self.train_step_counter)
                tf.summary.scalar(
                    name='rewards_mean', data=tf.reduce_mean(rewards), step=self.train_step_counter)
