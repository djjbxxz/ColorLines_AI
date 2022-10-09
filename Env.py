from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment, tf_py_environment, tf_environment
from tf_agents.specs import array_spec, tensor_spec
import numpy as np
import config
from utils import platform
# reduce_gpu_memory_usage()
if platform() == "Windows":
    import sys
    sys.path.append(
        R'D:\source\repos\ColorLine_DataGeneration\x64\export_pybind')
    import gen_colorline_data_tensorflow as gen_colorline_data
else:
    import gen_colorline_data_tensorflow as gen_colorline_data


class ColorLineEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=config.POTENTIAL_MOVE_NUM-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(
            config.BOARD_SIZE, config.BOARD_SIZE, config.INPUT_CHANNEL_SIZE), dtype=np.int32, minimum=0, maximum=1, name='observation')

        self._state = gen_colorline_data.get_random_start(
            config.LINED_NUM, config.FILL_RATIO)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = gen_colorline_data.get_random_start(
            config.LINED_NUM, config.FILL_RATIO)

        self._episode_ended = False
        self._score = 0
        return ts.restart(self.get_observation())

    def get_observation(self):
        '''
        Get observation based on current state
        '''
        return np.array(gen_colorline_data._994_to_9928(self._state), copy=False)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()
        # Agent acts here
        result = gen_colorline_data.rule(self._state, action)
        if result == -1:
            self._episode_ended = True
            self.last_step_reward = 0
        else:
            self._score += result
            self.last_step_reward = result
        if self._episode_ended:
            return ts.termination(observation=self.get_observation(), reward=0)
        else:
            return ts.transition(observation=self.get_observation(), reward=result, discount=config.REWARD_DISCOUNT)


if __name__ == '__main__':
    batch_size = 3
    import tensorflow as tf
    from tf_agents.environments import py_environment, tf_py_environment, tf_environment, batched_py_environment
    from show import show_Board
    env = tf_py_environment.TFPyEnvironment(
        batched_py_environment.BatchedPyEnvironment(
            [ColorLineEnv()for _ in range(batch_size)],
            multithreading=False
        ),
        isolation=False
    )
    time_step = env.reset()
    for i in range(batch_size):
        show_Board(time_step.observation[i, :])
    action = tf.zeros(shape=(batch_size,), dtype=tf.int32)
    next_time_step = env.step(action)
    for i in range(batch_size):
        show_Board(next_time_step.observation[i, :])
    a = 3
