import os
import yaml
import argparse
from datetime import datetime
from Env import ColorLineEnv
from sacd.agent import SacdAgent
from utils import reduce_gpu_memory_usage,suppress_warning
suppress_warning()
reduce_gpu_memory_usage()
def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = ColorLineEnv()
    test_env = ColorLineEnv()

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    if args.load != '':
        agent.load_models(args.load)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true',default=False)
    parser.add_argument('--env_id', type=str, default='ColorLine')
    parser.add_argument('--load', type=str, default=R'')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
