# Game configuration

#simiplifed
COLOR_NUM = 4
BOARD_SIZE = 5
COMING_CHESS_NUM = 2
MIN_ELEMINATABLE_NUM = 4
EACH_CHESS_ELEMINATED_REWARD = 2

#original
# COLOR_NUM = 7
# BOARD_SIZE = 9
# COMING_CHESS_NUM = 3
# MIN_ELEMINATABLE_NUM = 5
# EACH_CHESS_ELEMINATED_REWARD = 2




CHESS_NUM = BOARD_SIZE*BOARD_SIZE
POTENTIAL_MOVE_NUM = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE
INPUT_CHANNEL_SIZE = COLOR_NUM * (1+COMING_CHESS_NUM)



NN_INPUT_SHAPE = (None, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNEL_SIZE)
# Data generation configuration

LINED_NUM = 1
FILL_RATIO = 0.1

# Env configuration
REWARD_DISCOUNT = 0.99
ILLIGAL_MOVE_REWARD_PUNISH = -2
REWARD_SCALE = 1.0
