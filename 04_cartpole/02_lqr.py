"""
LQR (report's rho=10 setup).

Change MODE to "balance" or "step" and run: python 02_lqr.py
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from common import linear_model, run


MODE = "balance"
RHO = 10.0
# RHO = 1.0

A, B = linear_model()
C = np.array([[1.0, 0.0, 0.0, 0.0]])
Q = RHO * C.T @ C
R = np.array([[1.0]])

P = solve_continuous_are(A, B, Q, R)
K = (np.linalg.solve(R, B.T @ P))[0]

print("LQR K:", np.round(K, 4))
run(K, mode=MODE, name="02_lqr")
