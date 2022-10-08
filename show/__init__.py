try:
    from visualization import show_Board,show_episode,show_traj,show_transition
except ImportError as e:
    show_Board,show_episode,show_traj,show_transition=[None]*4
    print('cannot show images!')