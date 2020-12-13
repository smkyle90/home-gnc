#!/usr/bin/env python3
import time

import numpy as np
from rssi import RSSI_Scan


def expected_distance(rtr_loc, est_loc):
    """
    Calculate the expected distance based on the known router
    location and the location estimate.
    """
    return np.linalg.norm(rtr_loc - est_loc)


def scan_networks(interface="wlp5s0", sudo=True):
    # Scan for dataM
    r = RSSI_Scan(interface)
    s = r.getRawNetworkScan(sudo)
    s = s["output"].decode("utf-8")
    net_data = r.formatCells(s)

    if not net_data:
        return []

    return net_data


def expected_rssi(rssi0, env_factor, x0, y0, x, y):
    """
    Expected RSSI distance.
    """
    rtr_loc = np.array([[x0], [y0]])
    est_loc = np.array([[x], [y]])

    try:
        return rssi0 - 10 * env_factor * np.log10(expected_distance(rtr_loc, est_loc))
    except Exception:
        return rssi0


def calibrate(ssid):
    while True:
        net_data = scan_networks()
        try:
            for net in net_data:
                if net["ssid"] == ssid:
                    print(net["signal"])
        except Exception as e:
            print("Logger: Exception: {}".format(e))
        time.sleep(2)
