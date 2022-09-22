from collections import deque
import numpy as np


def update_params(optim, loss, var_list, tape):
    optim.minimize(loss, var_list, tape=tape)


def disable_gradients(network):
    # Disable calculations of gradients.
    network.trainable = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
