from utils import convert_6561_to_coord, isDebug, parse_9928_to_gamemap_next3, parse_9928_to_gamemap_next3_WHC, isDebug, parse_9928_to_gamemap_next3_CWH
import config
import numpy as np
import matplotlib.patches as mpathes
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['toolbar'] = 'None'
if isDebug():
    plt.ion()
Index2Color = {0: '#FFFFFF', 1: '#009C08', 2: '#ED1C24', 3: '#000084',
               4: '#B5A518', 5: '#18C6F7', 6: '#C618C6', 7: '#943100'}

scale = 0.5


def show_traj(experience, index=0):
    assert hasattr(
        experience, 'action'), 'experience has no attribute `action`'
    assert hasattr(
        experience, 'reward'), 'experience has no attribute `reward`'
    assert hasattr(
        experience, 'observation'), 'experience has no attribute `observation`'
    ob = experience.observation['observations']
    if index == -1:  # show successful move
        r = experience.reward.numpy()
        if np.sum(r) > 0:
            index = np.where(r)[0]
        else:
            print('There is no successful move lead to score, showing random traj')
            index = np.random.randint(config.batch_size)
    show_Board(ob[index, 0, :])
    action = experience.action[index, 0].numpy()
    action = convert_6561_to_coord(action)
    reward = experience.reward[index, 0].numpy()
    print(f'Move: {action}')
    show_Board(ob[index, 1, :])
    print(f'Reward: {reward}')


def show_transition(experience):
    batch_size = experience.action.numpy().shape[0]
    show_index = np.random.randint(batch_size)
    show_Board(experience.observation['observations']
               [show_index, 0, :], real_move=experience.action[show_index, 0].numpy())
    show_Board(experience.observation['observations'][show_index, 1, :])
    print(f"showing transition for index: {show_index}")
    pass


def show_Board(observation, real_move=None, show_text: bool = True):
    if hasattr(observation, 'numpy'):
        observation = observation.numpy()

    game_map, coming_chess = parse_9928_to_gamemap_next3(observation)

    if show_text:
        print(f"game_map:\n{game_map}")
        print(f"next_three:{coming_chess}")
    board = Board(game_map, coming_chess, real_move=real_move)
    return board.fig, board.ax


def show_episode(episode: list):
    '''`episode`should be a list of Trajectory'''
    assert isinstance(episode, list), f"arg type {type(episode)} is invalid!"
    print(f"showing a episode of {len(episode)}")
    if not isDebug():
        plt.ioff()
    Episode(episode)


class Board:

    def __init__(self, game_map, next_three=None, score=None, real_move=None, fig=None, ax=None, call_show=True):
        self.game_map = game_map
        self.next_three = next_three
        self.score = score
        self.real_move = real_move
        if fig == None and ax == None:
            self.fig, self.ax = self.get_ax()
        else:
            self.fig = fig
            self.ax = ax
        self.show()
        if call_show:
            plt.show()

    @staticmethod
    def get_ax():
        fig, ax = plt.subplots(
            figsize=(9*scale*config.BOARD_SIZE/8, 10*scale*config.BOARD_SIZE/8))

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.subplots_adjust(
            left=0, bottom=-10/(config.BOARD_SIZE+1)+1, right=10/config.BOARD_SIZE, top=1)
        Board.set_ax(ax)
        return fig, ax

    @staticmethod
    def set_ax(ax):
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.invert_yaxis()
        ax.yaxis.set_ticks_position('left')

    def show(self):
        self.drawgrid()
        self.drawboardChess()
        self.drawmove()
        self.drawnext3()
        self.drawscore()
        self.drawpossiblemove()

    def drawgrid(self):
        for i in range(config.BOARD_SIZE):
            self.ax.hlines(y=(i+1)/10, xmin=0,
                           xmax=config.BOARD_SIZE/10, color='black')
            self.ax.vlines(x=(i+1)/10, ymin=0.1,
                           ymax=(config.BOARD_SIZE+1)/10, color='black')

    def drawboardChess(self):
        game_map = self.game_map
        for i in range(config.BOARD_SIZE):
            for j in range(config.BOARD_SIZE):
                if game_map[i][j] != 0:
                    self.plotFilledCircle(
                        [i+1, j], Index2Color[game_map[i][j]], self.ax)

    def plotmove(self, start, end, text):
        self.ax.add_patch(self.plotRetangle(start))
        self.ax.add_patch(self.plotRetangle(end))
        self.plotLine_index(start, end, '->', text)

    def drawmove(self):
        if isinstance(self.real_move, np.ndarray):
            if self.real_move.size == 1:
                self.real_move = convert_6561_to_coord(self.real_move)
            t = np.reshape(self.real_move, (2, 2))
            start = t[0]
            end = t[1]
        elif isinstance(self.real_move, Board):
            start = self.real_move.last_move[0]
            end = self.real_move.last_move[1]
        elif isinstance(self.real_move, (np.int, np.int32, int)):
            self.real_move = convert_6561_to_coord(self.real_move)
            t = np.reshape(self.real_move, (2, 2))
            start = t[0]
            end = t[1]
        else:
            return
        self.plotmove(start, end, '')

    def plotLine_index(self, start, end, shape, text):
        arrow_args = dict(arrowstyle=shape)
        self.ax.annotate(text, xy=[(end[1]+0.5)/10, (8-end[0]+0.5)/10], xycoords='axes fraction', color='white',
                         xytext=[(start[1]+0.5)/10, (8-start[0]+0.5)/10],  arrowprops=arrow_args)

    def drawnext3(self):
        # draw coming chess grid
        if self.next_three is None:
            return
        left_margin = (config.BOARD_SIZE-config.COMING_CHESS_NUM)/2/10
        for i in range(config.COMING_CHESS_NUM+1):
            self.ax.vlines(x=left_margin+i/10, ymin=0, ymax=0.1, color='black')
        comingcolor = self.next_three

        for i in range(config.COMING_CHESS_NUM):
            self.plotFilledCircle(
                [0, left_margin*10+i], Index2Color[comingcolor[i]], self.ax)

    def drawscore(self):
        if self.score == None:
            return
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax.text((config.BOARD_SIZE-1.5)/10, 0.06, 'score:' +
                     str(self.score), bbox=props)

    @staticmethod
    def plotFilledCircle(xy, color, ax):
        xy = ((xy[1]+0.5)/10, (xy[0]+0.5)/10)
        circle = plt.Circle(xy, 0.8/20, color=color)
        ax.add_patch(circle)

    @staticmethod
    def plotRetangle(xy):
        return mpathes.Rectangle([((xy[1]+0.05)/10),
                                  (xy[0]+0.05+1)/10],
                                 0.09,
                                 0.09,
                                 color='r', fill=False)

    def drawpossiblemove(self):

        pass


class Episode(Board):

    def __init__(self, episode: list):
        self.fig, self.ax = self.get_ax()
        self.episode = episode
        self.len = len(episode)
        self.show_index = 0
        self.draw_button()
        self.flush(True)

    def draw_button(self):
        ax1 = plt.axes([0.01, 0.91, 0.12, 0.075])
        ax2 = plt.axes([0.18, 0.91, 0.12, 0.075])
        self.button1 = Button(ax1, "last")
        self.button1.on_clicked(self.get_last)
        self.button2 = Button(ax2, "next")
        self.button2.on_clicked(self.get_next)

    def flush(self, call_show):
        traj = self.episode[self.show_index]
        game_map, next_three = parse_9928_to_gamemap_next3_CWH(
            traj[0])
        score = traj[2]
        real_move = traj[1]
        super().__init__(game_map, next_three, score,
                         real_move, self.fig, self.ax, call_show=call_show)

    def clear_ax(self):
        plt.axes(self.ax)

        plt.cla()
        self.set_ax(self.ax)  # 清空 axes后重新设置axes 的属性

    def get_last(self, event):
        if self.show_index == 0:
            return
        self.clear_ax()
        self.show_index -= 1
        self.flush(False)
        plt.draw()

    def get_next(self, event):
        if self.show_index == self.len-1:
            return
        self.clear_ax()
        self.show_index += 1
        self.flush(False)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':

    a = np.zeros(shape=(config.BOARD_SIZE, config.BOARD_SIZE,
                        config.INPUT_CHANNEL_SIZE), dtype=np.int32)
    a[:, 1, 0] = 1
    a[:, 2, 1] = 1

    # a[:,:,4]=1
    a[:, :, -1] = 1
    show_Board(a)
    b = 3
