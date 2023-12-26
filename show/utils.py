import config
import numpy as np

def isDebug():
    import sys
    return True if sys.gettrace() else False


def convert_6561_to_coord(densed):
    '''arg: index of move (6561), return move by coordinate [1,3,5,2]]'''
    coord = np.zeros(shape=(4,), dtype=np.int32)
    for i in range(4):
        coord[3-i] = densed % config.BOARD_SIZE
        densed = densed / config.BOARD_SIZE
    return coord


def convert_coord_to_6561(coord):
    count = 0
    for i in range(4):
        count += config.BOARD_SIZE**(3-i)*coord[i]
    return count


def parse_9928_to_gamemap_next3_WHC(observation):
    assert observation.shape in [
        (config.BOARD_SIZE, config.BOARD_SIZE,  config.INPUT_CHANNEL_SIZE), (1, config.BOARD_SIZE, config.BOARD_SIZE,  config.INPUT_CHANNEL_SIZE)], f"{observation.shape} is invalid!"
    observation = np.reshape(
        observation, (config.BOARD_SIZE, config.BOARD_SIZE,  config.INPUT_CHANNEL_SIZE))
    game_map = np.zeros(
        shape=(config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.int32)
    for x in range(config.BOARD_SIZE):
        for y in range(config.BOARD_SIZE):
            color_one_hot = observation[x, y, :config.COLOR_NUM]
            if color_one_hot.max() == 0:
                continue
            assert np.sum(
                color_one_hot) < 2, f"color:{color_one_hot} is invalid!"
            color = np.argmax(color_one_hot)+1
            game_map[x, y] = color
    next_three_one_hot = observation[0, 0, config.COLOR_NUM:]
    next_three_one_hot = np.reshape(
        next_three_one_hot, (config.COMING_CHESS_NUM, config.COLOR_NUM))
    next_three = np.zeros(shape=(config.COMING_CHESS_NUM,), dtype=np.int32)
    for index, one_hot in enumerate(next_three_one_hot):
        color = np.argmax(one_hot)+1
        next_three[index] = color
    return game_map, next_three


def parse_9928_to_gamemap_next3_CWH(observation):
    assert_shape = (config.INPUT_CHANNEL_SIZE,
                    config.BOARD_SIZE, config.BOARD_SIZE)
    assert observation.shape in [
        assert_shape, (1, *assert_shape)], f"{observation.shape} is invalid!"
    observation = np.reshape(
        observation, (config.INPUT_CHANNEL_SIZE, config.BOARD_SIZE, config.BOARD_SIZE))
    game_map = np.zeros(
        shape=(config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.int32)
    for x in range(config.BOARD_SIZE):
        for y in range(config.BOARD_SIZE):
            color_one_hot = observation[:config.COLOR_NUM, x, y]
            if color_one_hot.max() == 0:
                continue
            assert np.sum(
                color_one_hot) == 1, f"color:{color_one_hot} is invalid!"
            color = np.argmax(color_one_hot)+1
            game_map[x, y] = color
    next_three_one_hot = observation[config.COLOR_NUM:, 0, 0]
    next_three_one_hot = np.reshape(
        next_three_one_hot, (config.COMING_CHESS_NUM, config.COLOR_NUM))
    next_three = np.zeros(shape=(config.COMING_CHESS_NUM,), dtype=np.int32)
    for index, one_hot in enumerate(next_three_one_hot):
        color = np.argmax(one_hot)+1
        next_three[index] = color
    return game_map, next_three


def parse_9928_to_gamemap_next3(observation):
    CWH_shape = (config.INPUT_CHANNEL_SIZE,
                 config.BOARD_SIZE, config.BOARD_SIZE)
    if observation.shape in [CWH_shape, (1, *CWH_shape)]:  # pytorch
        return parse_9928_to_gamemap_next3_CWH(observation)
    WHC_shape = (config.BOARD_SIZE, config.BOARD_SIZE,
                 config.INPUT_CHANNEL_SIZE)
    if observation.shape in [WHC_shape, (1, *WHC_shape)]:  # tensorflow
        return parse_9928_to_gamemap_next3_WHC(observation)


def convert_994_to_9928(game_map: np.ndarray) -> np.ndarray:
    new_game_map = np.zeros(
        shape=(config.BOARD_SIZE, config.BOARD_SIZE, config.INPUT_CHANNEL_SIZE), dtype=np.int32)
    for next_color in range(config.COMING_CHESS_NUM):
        color = game_map[0, 0, next_color+1]
        if color:
            new_game_map[:, :, config.COLOR_NUM+next_color*config.COLOR_NUM+color-1] = 1
    for x in range(config.BOARD_SIZE):
        for y in range(config.BOARD_SIZE):
            t = game_map[x, y, 0]
            if t != 0:
                new_game_map[x, y, t-1] = 1

    return new_game_map
