from Env import ColorLineEnv
from .visualization import Board

def get_game_window():
    return Board.get_ax()

def show(fig,ax,game_state):
    Board()


def main():
    fig,ax = get_game_window()
    env = ColorLineEnv()



if __name__ == '__main__':
    from visualization import show_Board
    from utils import *
    total_reward = 0.
    action=0
    env = ColorLineEnv()
    timestep = env.reset()
    ob = env.get_observation()
    show_Board(ob['observations'])
    while(True):
        timestep = env.step(action)
        show_Board(timestep[0])
        total_reward+=timestep[1]
        if timestep[2]:
            break
    print(f'Total reward:{total_reward}')
    a = 3