import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from lib.models import Unicycle

uni = Unicycle()

_T = 0
dt = 0.01

# uni_bar = copy.deepcopy(uni)

qn = np.zeros((3, 1))
ql = np.zeros((3, 1))
u = np.zeros((2, 1))

QN = []
QL = []
T = []

qd = np.array([[10], [10], [0]])

while _T < 10:
    # u = np.array([
    # 	[v + 0*np.random.normal()],
    # 	[omega + 0*np.random.normal()]
    # ])
    A = np.array(
        [
            [0, 0, -u[0, 0] * np.sin(ql[2, 0])],
            [0, 0, u[0, 0] * np.cos(ql[2, 0])],
            [0, 0, 0],
        ]
    )

    B = np.array([[np.cos(ql[2, 0]), 0], [np.sin(ql[2, 0]), 0], [0, 1],])

    K = signal.place_poles(A, B, [-1, -2, -3])

    u = -K.dot(qd - qn)
    G = np.array([[np.cos(qn[2, 0]), 0], [np.sin(qn[2, 0]), 0], [0, 1],])
    qn = qn + G.dot(u) * dt

    ql = ql + (A.dot(ql) + B.dot(u)) * dt

    print(qn)
    print(ql)

    QN.append(qn)
    QL.append(ql)
    T.append(_T)

    # uni.apply_control(u, dt)
    # A, B = uni_bar.linear_matrices(u)
    # print (uni.q)
    # uni_bar.q = uni_bar.q + (A.dot(uni_bar.q) + B.dot(u))*dt
    # print (uni_bar.q)
    _T += dt

plt.plot(np.array(T), np.block(QN).T - np.block(QL).T)
# plt.plot(np.array(T), np.block(QL).T)

plt.show()
