import os
import shutil
from tf_agents.utils import common
import yaml
import argparse
from datetime import datetime
from Env import ColorLineEnv
from agent import SacdAgent
from utils import isDebug, reduce_gpu_memory_usage, suppress_warning, get_current_pid
from sacd.model import *
from tf_agents.environments import tf_py_environment
import config as _config
import train_config
from Myreplay_buffer import Replay_buffer
from show import show_Board
import numpy as np
import contextlib
suppress_warning()
# reduce_gpu_memory_usage()

def _print_error(func):
    def wrapper(*args, **kwargs):
        if isDebug():
            return func(*args, **kwargs)
        else:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                return None
    return wrapper
    

@_print_error
def run(args):
    env = ColorLineEnv()
    test_env = ColorLineEnv()
    test_env = tf_py_environment.TFPyEnvironment(test_env)


    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'train_logs'if '' == args.logdir else args.logdir
    log_dir = os.path.join(log_dir, f'{time}')
    log_dir+=f'_{get_current_pid()}'
    if args.name:
        log_dir += f'_{args.name}'
    # Create the agent.

    agent = SacdAgent(
        save_dir=log_dir,
        env=env,
    )
    if not isDebug():
        agent.train = common.function(agent.train)
    replay_buffer = Replay_buffer(train_env=env, agent=agent,use_PER=args.PER or train_config.use_PER)

    if len(args.load) > 0:
        agent.load(
            os.path.dirname(args.load),
            os.path.basename(args.load)
        )

    def train_loop():
        replay_buffer.initial_fill_buffer(
            agent.collect_policy, train_config.init_fill)
        for i in range(train_config.train_steps):
            replay_buffer.collect()
            data, sample_info = replay_buffer.sample()
            if args.PER:
                loss_info = agent.train(data, weights=tf.cast(sample_info.priority[:, 0], tf.float32))
                replay_buffer.update_priorities(sample_info.key[:, 0], loss_info.extra.error)
            else:
                agent.train(data)
            agent.save_latest()

            # step = agent.train_step_counter.numpy()
            # print(
            #     f"{step}/{train_config.num_steps}, {np.round(step/train_config.num_steps*100,2)}%")
            agent.eval(test_env)

    if isDebug():
        train_loop()
    else:
        print(f'logging to : {log_dir}')
        test_summary_writer = tf.summary.create_file_writer(log_dir)
        shutil.copy('train_config.py', log_dir+'/train_config.py')
        shutil.copy('config.py', log_dir+'/config.py')
        if args.hide:
            with open(os.path.join(log_dir,'output.txt'), 'w') as f:
                with contextlib.redirect_stdout(f):
                    with test_summary_writer.as_default():
                        train_loop()
        else:
            with test_summary_writer.as_default():
                train_loop()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=R'')
    parser.add_argument('--logdir', type=str, default=R'')
    parser.add_argument('--name', type=str, default=R'',help='name of the run')
    parser.add_argument('--PER', action='store_true',default=True,help='enable prioritized experience replay')
    parser.add_argument('--hide', action='store_true',default=False,help='all output will be redirect to log dir')

    args = parser.parse_args()
    run(args)
