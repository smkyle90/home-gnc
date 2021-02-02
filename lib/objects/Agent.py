import time

import numpy as np


class Agent:
    def __init__(self, timestamp=None):

        if timestamp is None:
            self.timestamp = time.time()
        else:
            self.timestamp = timestamp

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if k[:1] != "_"}
