from abc import ABC, abstractmethod
import os
import numpy as np
import tensorflow as tf
import config
from sacd.utils import RunningMeanStats
from reverb_memory import Memory

class BaseAgent(ABC,object):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0):
        super().__init__()
        self.env = env
        self.test_env = test_env

        # Set seed.
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = tf.device('gpu')

        self.memory = Memory()

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = tf.summary.create_file_writer(self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)
        
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    @abstractmethod
    def learn(self):
        pass

    def train_episode(self):
        self.memory.clear_cache()
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0
        done = False
        reward = np.array(0,dtype = np.float32)
        state = self.env.reset_get_naked_state()

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.start_steps > self.steps:
                action = np.random.randint(config.POTENTIAL_MOVE_NUM)
            else:
                action = self.explore(state)

            action = np.array(action,dtype = np.int32)
            self.memory.append(state, action, reward, done)

            next_state, reward, done = self.env.step(action)

            # To calculate efficiently, set priority=max_priority here.

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

        self.memory.append(state, action, reward, done)


        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.steps)

            print(f'Episode: {self.episodes:<4}  '
                  f'Episode steps: {episode_steps:<4}  '
                  f'Return: {episode_return:<5.1f}')

    def evaluate(self, collect_all_timestep=False):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0
        time_steps = []
        while True:
            state = self.test_env.reset_get_naked_state()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                action = self.exploit(state)
                next_state, reward, done = self.test_env.step(action)
                time_steps.append([state, action, reward])
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))
        with self.writer.as_default(self.learning_steps):
            tf.summary.scalar('reward/test', mean_return)
            self.writer.flush()
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)
        if collect_all_timestep:
            return time_steps

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
