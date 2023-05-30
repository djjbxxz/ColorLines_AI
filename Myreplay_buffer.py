from tf_agents.policies import py_tf_eager_policy
import config

import threading
from Env import ColorLineEnv
from agent import SacdAgent
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.specs import tensor_spec
import reverb
import tensorflow as tf
class Replay_buffer():

    def __init__(self, train_env: ColorLineEnv, agent: SacdAgent, update_interval:int) -> None:
        self.train_env = train_env
        self.agent = agent
        self.get_replay_buffer(update_interval)
        self._last_timestep = None

    def initial_fill_buffer(self,policy:RandomTFPolicy,num:int):
        collect_driver = py_driver.PyDriver(
            self.train_env,
            py_tf_eager_policy.PyTFEagerPolicy(policy,use_tf_function=True),
            max_steps=num,
            observers=[self.rb_observer]
        )
        collect_driver.run(self.train_env.reset())

    def collect(self,num_batch):
        if self._last_timestep == None:
            self._last_timestep = self.train_env.reset()
        for _ in range(num_batch):
            time_step, _ = self._driver.run(time_step=self._last_timestep)
            self._last_timestep = time_step

    @property
    def iterator(self) -> iter:
        return self._iter

    @property
    def collect_driver(self) -> dynamic_step_driver.DynamicStepDriver:
        return self._driver

    def get_replay_buffer(self,update_interval:int):
        agent = self.agent

        table_name = 'table'
        replay_buffer_signature = tensor_spec.from_spec(
            agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size=config.replay_buffer_max_length,
            sampler=reverb.selectors.Prioritized(0.6),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)

        
        collect_driver = py_driver.PyDriver(
            self.train_env,
            py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy,use_tf_function=True),
            observers=[self.rb_observer],
            max_steps=update_interval
        )

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=config.batch_size,
            num_steps=2).prefetch(3)
        self._iter = iter(dataset)
        self._driver = collect_driver
        self.replay_buffer = replay_buffer

    def update_priorities(self,keys,priorities):
        #return np.clip((p + self.eps) ** self.alpha, self.min_pa, self.max_pa)
        priorities = (priorities+0.01)**0.6
        priorities = tf.clip_by_value(priorities,0,1)
        priorities = tf.cast(priorities,tf.float64)
        self.replay_buffer.update_priorities(keys,priorities)
