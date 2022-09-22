from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
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


class ColorLineEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=config.POTENTIAL_MOVE_NUM-1, name='action')
        self._observation_spec = {'observations': array_spec.BoundedArraySpec(shape=(config.INPUT_CHANNEL_SIZE, config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.int32, minimum=0, maximum=1, name='observation'),
                                  'legal_moves': array_spec.BoundedArraySpec(shape=(config.POTENTIAL_MOVE_NUM,), dtype=np.int32, minimum=0, maximum=1, name='legal_moves')}
        self._state = gen_colorline_data.get_random_start(
            config.LINED_NUM, config.FILL_RATIO)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # def reward_spec(self):
    #     return self._reward_spec

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
        return {'observations': np.array(gen_colorline_data._994_to_9928(self._state), copy=False),
                'legal_moves': np.array(gen_colorline_data.get_valid_mask(self._state), copy=False).astype(np.int32)}

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
            reward = self._score
            return ts.termination(observation=self.get_observation(), reward=0)
        else:
            return ts.transition(observation=self.get_observation(), reward=result, discount=config.REWARD_DISCOUNT)

    def reset_get_naked_state(self):
        val = super().reset()
        return self.getstatefromob(val)

    def step(self, a):
        val = super().step(a)
        return self.getstatefromob(val), val.reward, self._episode_ended

    @staticmethod
    def getstatefromob(val):
        return val.observation['observations']

if __name__ == '__main__':
    from visualization import show_Board
    from utils import *
    total_reward = 0.
    action=0
    env = ColorLineEnv()
    timestep = env.reset()
    ob = env.get_observation()
    show_Board(ob['observations'])
    while(True):
        timestep = env.step(action)
        show_Board(timestep[0])
        total_reward+=timestep[1]
        if timestep[2]:
            break
    print(f'Total reward:{total_reward}')
    a = 3
