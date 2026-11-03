"""
Full-order observer + pole-placement state feedback.

Only cart position x is measured by the observer.
The observer estimates cart velocity, pole angle, and pole angular velocity.
Run: python 03_estimator.py
"""

import numpy as np
from scipy.signal import place_poles
from common import linear_model, run


MODE = "balance"

A, B = linear_model()
C = np.array([[1.0, 0.0, 0.0, 0.0]])

# Controller
poles = [-0.23 + 0.2346j, -0.23 - 0.2346j, -2.30, -2.53]
K = place_poles(A, B, poles).gain_matrix[0]

# Observer
observer_poles = [5 * pole for pole in poles]
L = place_poles(A.T, C.T, observer_poles).gain_matrix.T[:, 0]

print("Controller K:", np.round(K, 4))
print("Observer L:", np.round(L, 4))

run(K, mode=MODE, L=L, name="03_estimator")
