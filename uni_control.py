import numpy as np

from lib.funcs import feedback_linearisation_controller

q0 = np.array([[20], [0], [np.pi / 3]])
qf = np.array([[0], [0], [0]])

u_ff, q_path = feedback_linearisation_controller(q0, qf, T_max=100, vd=-1.0)


theta0 = q0[2, 0]
thetaf = qf[2, 0]

dq = (qf - q0)[:2, 0]
d_theta = np.arctan2(dq[1], dq[0])
