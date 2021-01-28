import numpy as np


class Obstacle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def as_vec(self):
        return np.array([self.x, self.y, self.r]).reshape(-1, 1)
