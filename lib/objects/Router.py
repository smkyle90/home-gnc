#!/usr/bin/env python3

import numpy as np

from ..funcs import expected_distance


class Router:
    def __init__(self, name, x0, y0, rssi0, env_factor=3, rssi_var=2):
        self.name = name
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.rssi0 = rssi0
        self.rssi_var = rssi_var
        self.env_factor = env_factor

    def as_vec(self):
        return np.array([self.rssi0, self.env_factor, self.x0, self.y0]).reshape(-1, 1)

    def loc(self):
        return np.array([[self.x0], [self.y0]])

    def expected_rssi(self, x, y):
        est_loc = np.array([[x], [y]])
        try:
            return self.rssi0 - 10 * self.env_factor * np.log10(
                expected_distance(self.loc(), est_loc)
            )
        except Exception:
            return self.rssi0

    def expected_rssi_cov(self, x, y):
        est_loc = np.array([[x], [y]])

        dell_x = (
            -10
            * self.env_factor
            * (x - self.x0)
            / expected_distance(self.loc(), est_loc) ** 2
        )
        dell_y = (
            -10
            * self.env_factor
            * (y - self.y0)
            / expected_distance(self.loc(), est_loc) ** 2
        )

        return [dell_x, dell_y]

    def rssi_to_dist(self, rssi):
        return 10 ** ((self.rssi0 - rssi) / (10 * self.env_factor))
