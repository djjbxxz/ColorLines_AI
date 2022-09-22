from utils import suppress_warning,reduce_gpu_memory_usage
suppress_warning()
reduce_gpu_memory_usage()
from sacd.agent import SacdAgent
from Env import ColorLineEnv
# import tensorflow as tf
env = ColorLineEnv()
test_env = ColorLineEnv()
agent = SacdAgent(env,test_env,'1')
agent.run()