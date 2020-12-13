#!/usr/bin/env python3
import time

import numpy as np
from numpy.linalg import inv


class Person:
    def __init__(self, x, y, cov):
        self.x = x
        self.y = y
        self.cov = cov
        self.timestamp = time.time()

    def loc(self):
        return np.array([[self.x], [self.y]])

    def predict(self, process_covar):
        dt = time.time() - self.timestamp
        self.cov += dt ** 2 * process_covar
        self.timestamp = time.time()

    def update(self, network_data, router_loc):
        z = []
        z_hat = []
        H = []
        R = []

        for net in network_data:
            if net["ssid"] in router_loc:
                rtr_inf = router_loc[net["ssid"]]
                calc_dist = rtr_inf.rssi_to_dist(net["signal"])
                print(
                    "Logger: SSID: {}, RSSI: {} dB, Calc Distance: {} m.".format(
                        net["ssid"], net["signal"], round(calc_dist, 2)
                    )
                )
                z.append(net["signal"])
                z_hat.append(rtr_inf.expected_rssi(*self.loc()[:, 0]))
                H.append(rtr_inf.expected_rssi_cov(*self.loc()[:, 0]))
                R.append(rtr_inf.rssi_var ** 2)

        if z:
            try:
                z = np.array(z).reshape(-1, 1)
                z_hat = np.array(z_hat).reshape(-1, 1)
                H = np.array(H)
                R = np.diag(R)
                inn_meas = z - z_hat
                inn_covar = H.dot(self.cov).dot(H.T) + R
                kal_gain = self.cov.dot(H.T).dot(inv(inn_covar))
                new_state = self.loc() + kal_gain.dot(inn_meas)

                self.x = new_state[0, 0]
                self.y = new_state[1, 0]
                self.cov = (np.eye(2) - kal_gain.dot(H)).dot(self.cov)
            except Exception as e:
                print("Logger: Exception: {}".format(e))

        self.cov = (self.cov + self.cov.T) / 2

        return len(z)
