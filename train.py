import os
import shutil
import yaml
import argparse
from datetime import datetime
from Env import ColorLineEnv
from agent import SacdAgent
from utils import isDebug, reduce_gpu_memory_usage, suppress_warning
from sacd.model import *
from tf_agents.environments import tf_py_environment, batched_py_environment
import config as _config
import train_config
from Myreplay_buffer import Replay_buffer
from show import show_Board
import numpy as np
suppress_warning()
reduce_gpu_memory_usage()


def run(args):

    env =ColorLineEnv()
    test_env = ColorLineEnv()
    test_env = tf_py_environment.TFPyEnvironment(test_env)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'train_logs', f'{time}')

    # Create the agent.

    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    q_network = QNetwork(observation_spec, action_spec)
    policy_network = ActorNetwork(observation_spec)
    lr = train_config.lr
    agent = SacdAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        critic_network=q_network,
        actor_network=policy_network,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(lr),
        critic_optimizer1=tf.compat.v1.train.AdamOptimizer(lr),
        critic_optimizer2=tf.compat.v1.train.AdamOptimizer(lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(lr),
        save_dir=log_dir
    )
    replay_buffer = Replay_buffer(train_env=env, agent=agent,update_interval=train_config.update_interval)

    if len(args.load) > 0:
        agent.load(
            os.path.dirname(args.load),
            os.path.basename(args.load)
        )

    def train_loop():
        replay_buffer.initial_fill_buffer(
            agent._random_policy, train_config.init_fill)
        for i in range(train_config.num_steps):
            replay_buffer.collect(1)
            data, sample_info = next(replay_buffer.iterator)
            loss_info = agent.train(data,weights = tf.cast(sample_info.priority[:,0],tf.float32))
            replay_buffer.update_priorities(sample_info.key[:,0],loss_info.extra.error)
            step = agent.train_step_counter.numpy()
            print(
                f"{step}/{train_config.num_steps}, {np.round(step/train_config.num_steps*100,2)}%")
            agent.eval(test_env)

    if isDebug():
        train_loop()
    else:
        test_summary_writer = tf.summary.create_file_writer(log_dir)
        shutil.copy('train_config.py', log_dir+'/train_config.py')
        with test_summary_writer.as_default():
            train_loop()


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
