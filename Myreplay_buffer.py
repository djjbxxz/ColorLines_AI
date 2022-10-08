import config
import reverb


from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import threading
from Env import ColorLineEnv
from agent import SacdAgent
from tf_agents.drivers import dynamic_episode_driver


class Replay_buffer():

    def __init__(self, train_env: ColorLineEnv, agent: SacdAgent, multithread=False) -> None:
        self.train_env = train_env
        self.agent = agent
        self.get_replay_buffer()
        self.multithread = multithread

    def __thread_collect_data(self, collect_driver, train_env):
        assert self.multithread, 'No multithread mode'
        time_step = train_env.reset()
        while(True):
            time_step, _ = collect_driver.run(time_step)

    def collect(self, step):
        time_step = self.train_env.reset()
        for _ in range(step):
            time_step, _ = self.collect_driver.run(time_step)

    @property
    def iterator(self) -> iter:
        return self._iter

    @property
    def driver(self) -> py_driver.PyDriver:
        return self._driver

    def get_replay_buffer(self):
        agent = self.agent
        table_name = 'table_name'
        replay_buffer_signature = tensor_spec.from_spec(
            agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size=config.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            # rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
            #     samples_per_insert=15, min_size_to_sample=config.batch_size, error_buffer=config.batch_size*40),
            signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)

        collect_driver = py_driver.PyDriver(
            self.train_env,
            py_tf_eager_policy.PyTFEagerPolicy(
                agent.collect_policy,
                use_tf_function=True),
            [rb_observer],
            max_steps=config.collect_steps_per_iteration
        )
        # collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        #     self.train_env, agent.collect_policy, [rb_observer], num_episodes=2)
        self.__worker_thread = threading.Thread(target=self.__thread_collect_data,
                                                args=(collect_driver, self.train_env))
        self.__worker_thread.setDaemon(True)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=config.batch_size,
            num_steps=2).prefetch(3)
        self._iter = iter(dataset)
        self._driver = collect_driver
        self.client = replay_buffer.py_client
        self.replay_buffer = replay_buffer
        self.collect_driver = collect_driver

    def start_collect_data(self):
        self.__worker_thread.start()
