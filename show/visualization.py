import sys
sys.path.append('.')
from show.utils import convert_6561_to_coord, convert_994_to_9928, convert_coord_to_6561, isDebug, parse_9928_to_gamemap_next3, parse_9928_to_gamemap_next3_WHC, isDebug, parse_9928_to_gamemap_next3_CWH
from utils import isDebug
import config
import numpy as np
import os
import matplotlib.patches as mpathes
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
HEADLESS = os.environ.get('DISPLAY', '') == ''
if not HEADLESS:
    mpl.use('TkAgg')
    mpl.rcParams['toolbar'] = 'None'
    if isDebug():
        plt.ion()
Index2Color = {0: '#FFFFFF', 1: '#009C08', 2: '#ED1C24', 3: '#000084',
               4: '#B5A518', 5: '#18C6F7', 6: '#C618C6', 7: '#943100'}
path_color = '#01C0F0'
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


def show_Board(observation, real_move=None, show_text: bool = True,score=None):
    if hasattr(observation, 'numpy'):
        observation = observation.numpy()

    game_map, coming_chess = parse_9928_to_gamemap_next3(observation)

    if show_text:
        print(f"game_map:\n{game_map}")
        print(f"next_three:{coming_chess}")
        board = Board(game_map, coming_chess, real_move=real_move,call_show=False,score=score)
        if HEADLESS:
            plt.savefig('board.png', dpi=250)
            plt.close(board.fig)
            print('board.png saved due to HEADLESS mode is on')
        else:
            plt.show()
            plt.close(board.fig)
            return board.fig, board.ax
    
def save_traj(trajectories, path:str, show_path_num:int = 1, show_probs:bool = False, show_score:bool = True):
    '''`trajectories`should be a list of Trajectory'''
    if not os.path.exists(path):
        os.makedirs(path)
    assert isinstance(trajectories, list), f"arg type {type(trajectories)} is invalid!"
    score = 0
    for i, trajectory in enumerate(trajectories):
        game_map, coming_chess = parse_9928_to_gamemap_next3(trajectory.observation['observations'])
        real_move =trajectory.policy_info['sorted_probs_index'][:show_path_num] if trajectory.step_type[0].numpy() != 2 else None
        move_probs = trajectory.policy_info['sorted_probs'][:show_path_num] if trajectory.step_type[0].numpy() != 2 else None
        score+=int(trajectories[i-1].reward[0].numpy())if i!=0 else 0
        board = Board(game_map, coming_chess, real_move=real_move,score=score if show_score else None ,call_show=False, move_probs = move_probs if show_probs else None)
        plt.savefig(path + f'/{i}.png', dpi=250)
        plt.close(board.fig)

def show_episode(episode: list):
    '''`episode`should be a list of Trajectory'''
    assert isinstance(episode, list), f"arg type {type(episode)} is invalid!"
    print(f"showing a episode of {len(episode)}")
    if not isDebug():
        plt.ioff()
    Episode(episode)


class Board:

    def __init__(self, game_map, next_three=None, score=None, real_move=None, fig=None, ax=None, call_show=True, move_probs=None):
        self.game_map = game_map
        self.next_three = next_three
        self.score = score
        self.real_move = real_move
        self.move_probs = move_probs
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

    def plot_path(self, path,prob = None):
        def transform(x,y):
            return ((y+0.5)/10, (x+0.5+1)/10)
        path = [transform(x,y) for x,y in path]
        verts = path
        codes = [Path.MOVETO] + [Path.LINETO]*(len(path)-1)
        path = Path(verts, codes)
        patch = FancyArrowPatch(path=path,
                        arrowstyle='-|>',
                        mutation_scale=20,
                        lw=2,
                        color=path_color,
                        )
        if prob is not None:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            self.ax.text(verts[0][0],verts[0][1],
                    str(np.round(prob,1)),size=5, bbox=props)
        self.ax.add_patch(patch)
                

    def drawmove(self):
        if isinstance(self.real_move, np.ndarray):
            import gen_colorline_data_tensorflow as m
            for index, real_move in enumerate(self.real_move):
                points = m.get_path(self.game_map,real_move)
                path = [(point.x,point.y) for point in points]
                prob = self.move_probs[index] if self.move_probs is not None else None
                self.plot_path(path,prob)
        elif isinstance(self.real_move, Board):
            start = self.real_move.last_move[0]
            end = self.real_move.last_move[1]
            path = [(start[0],start[1]),(end[0],end[1])]
        elif isinstance(self.real_move, (np.int32, int)):
            import gen_colorline_data_tensorflow as m
            points = m.get_path(self.game_map,self.real_move)
            path = [(point.x,point.y) for point in points]
            self.plot_path(path)
        else:
            return

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
        self.ax.text((config.BOARD_SIZE-1.25)/10, 0.06, 'score:' +
                     str(self.score),size=9, bbox=props)

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
    a[1:, 1, 0] = 1
    a[1:, 2, 1] = 1

    # a[:,:,4]=1
    a[1:, :, -1] = 1
    
    game_map = [[1,0,0,2,0],
                [3,2,0,1,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0]]
    game_map = np.stack([game_map,game_map,game_map,game_map],axis=-1)
    game_map = convert_994_to_9928(game_map)
    show_Board(game_map,convert_coord_to_6561([1,1,4,2]))
    b = 3
