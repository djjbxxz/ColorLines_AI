import os
import yaml
import argparse
from datetime import datetime
from Env import ColorLineEnv
from agent import SacdAgent
from utils import reduce_gpu_memory_usage, suppress_warning
from sacd.model import *
from tf_agents.environments import py_environment, tf_py_environment, tf_environment, batched_py_environment
import config as _config
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from Myreplay_buffer import Replay_buffer
from show import show_Board
suppress_warning()
reduce_gpu_memory_usage()


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    # ColorLineEnv =tf_env = tf_py_environment.TFPyEnvironment(env)
    # env = ColorLineEnv()
    # env = tf_py_environment.TFPyEnvironment(env)
    env = tf_py_environment.TFPyEnvironment(
        batched_py_environment.BatchedPyEnvironment(
            [ColorLineEnv()for _ in range(_config.batch_size)],
            multithreading=False
        ),
        isolation=False
    )
    test_env = ColorLineEnv()
    # test_env = tf_py_environment.TFPyEnvironment(test_env)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.

    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    q_network = QNetwork(observation_spec, action_spec)
    policy_network = ActorNetwork(observation_spec)

    agent = SacdAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        critic_network=q_network,
        actor_network=policy_network,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),

    )

    replay_buffer = Replay_buffer(train_env=env, agent=agent)

    for i in range(3000):
        replay_buffer.collect()
        data, _ = next(replay_buffer.iterator)
        agent.train(data)
        print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true', default=False)
    parser.add_argument('--env_id', type=str, default='ColorLine')
    parser.add_argument('--load', type=str, default=R'')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
