#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np
import yaml
import zmq

from lib.funcs import calibrate, scan_networks
from lib.objects import Person, Router


def main(config):
    push_addr = config["STATE"]["ADDR"]
    push_port = int(config["STATE"]["PORT"])

    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    # Binds the socket to a predefined port on localhost
    socket.bind("tcp://{}:{}".format(push_addr, push_port))
    time.sleep(1)

    interface = config["INTERFACE"]
    sudo = config["SUDO"]
    router_loc = {rtr[0]: Router(*rtr[:-1]) for rtr in config["ROUTER_LOC"]}
    process_covar = config["PROC_COVAR"] ** 2 * np.eye(2)

    est_state = np.mean(
        np.block([rtr.loc() for rtr in router_loc.values()]), axis=1
    ).reshape(-1, 1)
    est_covar = np.diag([10.0 ** 2, 10.0 ** 2])

    scott = Person(est_state[0, 0], est_state[1, 0], est_covar)

    while True:
        scott.predict(process_covar)
        print("Logger: Predict Step Complete.")

        print("Logger: Scanning for network data.")
        net_data = scan_networks(interface, sudo)

        print("Logger: Network Data received.")

        n_meas = scott.update(net_data, router_loc)
        print("Logger: Update Step Complete using {} measurements.".format(n_meas))

        if n_meas < 3:
            print("Logger: System is unobservable. Number of measurements is too low.")

        # Publish the data
        print("Logger: Publishing data.")
        json_msg = json.dumps(
            {"loc": scott.loc().tolist(), "cov": scott.cov.tolist(), "n_meas": n_meas}
        )
        socket.send_json(json_msg)
        print("Logger: Data published")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State Estimate.")
    parser.add_argument(
        "path",
        metavar="path/to/config/file",
        type=str,
        help="Path to configuration file.",
    )

    args = parser.parse_args()
    with open(args.path, "r") as f:
        config = yaml.safe_load(f)

    main(config["NAVIGATION"])
