# Training configuration

num_iterations = 500000  # @param {type:"integer"}

collect_steps_per_iteration = 10  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 256  # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}
log_interval = 100  # @param {type:"integer"}

eval_interval = 5000  # @param {type:"integer"}
save_interval = 1000

epsilon_greedy = 0.1     # 0.0 ~ 1.0

logdir = "logs/2"
checkpoint_dir = 'models/test_model1'

eval_max_episodes = 2
eval_max_steps = 200

gradient_step = 4 #train at every collect step
# Game configuration

COLOR_NUM = 4
BOARD_SIZE = 5
COMING_CHESS_NUM = 2
MIN_ELEMINATABLE_NUM = 4
EACH_CHESS_ELEMINATED_REWARD = 2
CHESS_NUM = BOARD_SIZE*BOARD_SIZE
POTENTIAL_MOVE_NUM = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE
INPUT_CHANNEL_SIZE = COLOR_NUM * (1+COMING_CHESS_NUM)



NN_INPUT_SHAPE = (None, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNEL_SIZE)
# Data generation configuration

LINED_NUM = 1
FILL_RATIO = 0.1

# Env configuration
REWARD_DISCOUNT = 1.
ILLIGAL_MOVE_REWARD_PUNISH = -2
