import numpy as np


class Arena:
    def __init__(self, x_min, x_max, dx, y_min, y_max, dy):
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.y_min = y_min
        self.y_max = y_max
        self.dy = dy

        x_vec = np.arange(x_min - dx, x_max + 2 * dx, dx)
        y_vec = np.arange(y_min - dy, y_max + 2 * dy, dy)
        self.x_mesh, self.y_mesh = np.meshgrid(x_vec, y_vec)
