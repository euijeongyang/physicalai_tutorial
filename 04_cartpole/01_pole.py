"""
Pole placement (report's state feedback controller).

Change MODE to "off", "balance", or "step" and run: python 01_pole.py
"""

import numpy as np
from scipy.signal import place_poles
from common import linear_model, run

# MODE = "off"
MODE = "balance"
# MODE = "step"

A, B = linear_model()

# Desired closed-loop poles
poles = [-0.23 + 0.2346j, -0.23 - 0.2346j, -2.30, -2.53]
K = place_poles(A, B, poles).gain_matrix[0]

print("Pole placement K:", np.round(K, 4))
run(K, mode=MODE, name="01_pole")
