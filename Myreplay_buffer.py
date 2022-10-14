import config

import threading
from Env import ColorLineEnv
from agent import SacdAgent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.random_tf_policy import RandomTFPolicy


class Replay_buffer():

    def __init__(self, train_env: ColorLineEnv, agent: SacdAgent, multithread=False) -> None:
        self.train_env = train_env
        self.agent = agent
        self.multithread = multithread
        self.get_replay_buffer()
        self._last_timestep = None

    def initial_fill_buffer(self,policy:RandomTFPolicy,num:int):
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            policy,
            num_steps=num*2,
            observers=[self.replay_buffer.add_batch]
        )
        collect_driver.run()

    def __thread_collect_data(self, collect_driver, train_env):
        assert self.multithread, 'No multithread mode'
        time_step = train_env.reset()
        while(True):
            time_step, _ = collect_driver.run(time_step)

    def collect(self,num_batch):
        for _ in range(num_batch):
            time_step, _ = self._driver.run(time_step=self._last_timestep)
            self._last_timestep = time_step

    @property
    def iterator(self) -> iter:
        return self._iter

    @property
    def collect_driver(self) -> dynamic_step_driver.DynamicStepDriver:
        return self._driver

    def get_replay_buffer(self):
        agent = self.agent
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=config.gradient_step,
            max_length=config.replay_buffer_max_length)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            agent.collect_policy,
            num_steps=2,
            observers=[replay_buffer.add_batch]
        )

        if self.multithread:
            self.__worker_thread = threading.Thread(target=self.__thread_collect_data,
                                                    args=(collect_driver, self.train_env))
            self.__worker_thread.setDaemon(True)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=config.batch_size,
            num_steps=2).prefetch(3)
        self._iter = iter(dataset)
        self._driver = collect_driver
        self.replay_buffer = replay_buffer

    def start_collect_data(self):
        self.__worker_thread.start()
