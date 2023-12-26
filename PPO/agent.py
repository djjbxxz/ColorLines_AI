import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from keras.activations import relu
import time
import numpy as np
from .base import PPOKLPenaltyAgent
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from .model import ValueNetwork, ActorNetwork
from .Env import ColorLineEnv, observation_and_action_constraint_splitter
from .utils import Checkpoint, Logger
from .replay_buffer_m import Replay_buffer


class PPO_Agent(PPOKLPenaltyAgent):
    def __init__(self,
                 train_env: ColorLineEnv,
                 test_env: ColorLineEnv,
                 eval_mode=False,
                 eval_interval: int = 3000,
                 eval_episodes: int = 30,
                 save_interval: int = 10000,
                 log_interval: int = 250,
                 batch_size: int = 64,
                 buffer_length: int = 100000,
                 learning_rate: float = 0.00025,
                 num_epochs: int = 10,
                 logdir: str = os.path.join(
                     'runs/ppo', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 name="",
                 **kwargs,
                 ):
        super().__init__(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            actor_net=ActorNetwork(None, train_env.action_spec(
            ), observation_and_action_constraint_splitter),
            value_net=ValueNetwork(None, train_env.action_spec(
            ), observation_and_action_constraint_splitter),
            num_epochs=num_epochs,
            optimizer=Adam(learning_rate=self.get_lr),
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=None,
            name=None,
            **kwargs,
        )
        self._learning_rate = learning_rate
        self.eval_mode = eval_mode
        self._train_env = train_env
        self._test_env = test_env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.log_dir = logdir if name == ""else logdir+f"_{name}"
        self._ckp = Checkpoint(
            dir=self.log_dir,
            save_interval=save_interval,
            policy=self.policy,
            step_counter=self.train_step_counter
        )
        self.load = self._ckp.load
        if not eval_mode:
            self.logger = Logger(logdir=self.log_dir)
            self.best_eval_return = -np.inf

            self.replay_buffer = Replay_buffer(
                train_env,
                self.collect_data_spec,
                self.collect_policy,
                maxsize=buffer_length,
                batch_size=batch_size)

            self.save = self._ckp.save_latest

    def run(self, num_iterations):
        for i in range(num_iterations):

            # Collect a few steps and save to the replay buffer.
            self.replay_buffer.collect()
            # Sample a batch of data from the buffer and update the agent's network.
            experience = self.replay_buffer.sample()
            # check_data(experience)
            self.train(experience)

            step = self.train_step_counter.numpy()
            self.log_hyper_params()

            # if step % self.log_interval == 0:
            # self.logger.log_data(
            #     step, train_loss, experience)

            if step % self.eval_interval == 0:
                r = self.eval()
                self.logger.eval_log(step, *r)

            self._ckp.save_latest()

    def log_hyper_params(self):
        with tf.name_scope('Hyperparams/'):
            tf.compat.v2.summary.scalar(
                name='learning_rate',
                data=self._learning_rate,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='entropy_regularization',
                data=self._entropy_regularization,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='value_pred_loss_coef',
                data=self._value_pred_loss_coef,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='policy_l2_reg',
                data=self._policy_l2_reg,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='value_function_l2_reg',
                data=self._value_function_l2_reg,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='shared_vars_l2_reg',
                data=self._shared_vars_l2_reg,
                step=self.train_step_counter)

    def useGraphOptimization(self):
        if not hasattr(self, 'old_train'):
            self.old_train = self.train
        self.train = common.function(self.old_train)
        self.graph_enabled = True

    def eval(self):
        observers = [
            tf_metrics.AverageEpisodeLengthMetric(),
            tf_metrics.AverageReturnMetric()
        ]
        # use policy to py_policy
        results = metric_utils.compute(observers,
                                       self._test_env,
                                       py_tf_eager_policy.PyTFEagerPolicy(
                                           self.policy, use_tf_function=True),
                                       self.eval_episodes)
        avg_return = results['AverageReturn'].numpy()

        if not self.eval_mode and avg_return > self.best_eval_return:
            self.best_eval_return = avg_return
            self._ckp.save_best()

        return results.values()

    # def infer(self,env:Env, job: Job, observation:np.ndarray ,allow_idle: bool) -> list:
    #     '''infer GPU selection for a job, return a list of GPU id

    #     Parameters
    #     ----------
    #     env : Env
    #         The environment
    #     job : Job
    #         The job to be infered
    #     allow_idle : bool, optional
    #         Whether allow idle action when there is enough GPU, by default False

    #     Returns
    #     -------
    #     action : list
    #         A set of GPU satisfying the requested number of GPUs
    #     '''
    #     actions = []
    #     mask = self.get_valid_mask(
    #         env, job=job, allow_idle=allow_idle, action_cache=[])
    #     _input = observation_struct(observation=observation, action_mask=mask)
    #     for i in range(job.gpu_request):
    #         time_step = TimeStep(0, 0, 0, observation=_input)
    #         # time_step = tf.nest.map_structure(
    #         #     lambda x: tf.convert_to_tensor(x, tf.float64), time_step)
    #         action = self.policy.action(time_step).action.numpy()
    #         actions.append(action)
    #         if action == 32:
    #             return []  # idle action
    #         _input.observation[action] = 1
    #         _input.action_mask[action] = 0
    #     return actions

    def get_lr(self):
        return self._learning_rate
