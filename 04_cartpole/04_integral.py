"""07: Integral state feedback (report pages 27-29).

Run: python 07_integral.py
A constant disturbance begins at 5 s. Compare reference compensation vs integral control.
Uses the same scene.xml as lessons 04-06. Close the viewer to exit.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from scipy.signal import place_poles

from common import linear_model, OUTPUT_DIR


# Experiment settings
DURATION = 20.0
DISTURBANCE_START = 5.0
DISTURBANCE_FORCE = 0.05   # N; positive X direction
TARGET_POSITION = 1.0     # m

# Controller model from the report
A, B = linear_model()
C = np.array([[1.0, 0.0, 0.0, 0.0]])

poles = [-0.23 + 0.2346j, -0.23 - 0.2346j, -2.30, -2.53]

# Previous controller: u = -K*x + Nbar*r
K = place_poles(A, B, poles).gain_matrix[0]
Nbar = K[0]

# Integral controller: z_dot = x - r, u = -Kx*x - Ki*z
# Add one extra state (z), so we need one additional pole.
A_int = np.block([
    [A, np.zeros((4, 1))],
    [C, np.zeros((1, 1))],
])
B_int = np.vstack([B, [[0.0]]])
K_int = place_poles(A_int, B_int, poles + [-2.0]).gain_matrix[0]
Kx = K_int[:4]
Ki = K_int[4]

print("Reference compensation K:", np.round(K, 4))
print("Integral controller Kx:", np.round(Kx, 4))
print("Integral gain Ki:", round(Ki, 4))


def simulate(controller, show_viewer=False):
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
    data = mujoco.MjData(model)

    cart = model.joint("cart_joint")
    pole = model.joint("pole_joint")
    x_id = cart.qposadr[0]
    vx_id = cart.dofadr[0]
    angle_id = pole.qposadr[0]
    omega_id = pole.dofadr[0]
    motor_id = model.actuator("cart_motor").id

    integral_error = 0.0
    records = []
    next_print = 0.0

    viewer = None
    if show_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        with viewer.lock():
            viewer.cam.lookat[:] = [0.3, 0, 0.7]
            viewer.cam.distance = 3.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -12

    try:
        while data.time < DURATION:
            if viewer is not None and not viewer.is_running():
                break

            start = time.monotonic()
            x = data.qpos[x_id]
            state = np.array([x, data.qvel[vx_id], data.qpos[angle_id], data.qvel[omega_id]])

            if controller == "integral":
                force = -Kx @ state - Ki * integral_error
            else:
                force = -K @ state + Nbar * TARGET_POSITION

            force = float(np.clip(force, -30.0, 30.0))
            data.ctrl[motor_id] = force

            # External disturbance, separate from the controller's command
            disturbance = DISTURBANCE_FORCE if data.time >= DISTURBANCE_START else 0.0
            data.qfrc_applied[vx_id] = disturbance

            mujoco.mj_step(model, data)

            # Integrate position error. Keep the state bounded if force saturates.
            if controller == "integral" and -30.0 < force < 30.0:
                integral_error += (x - TARGET_POSITION) * model.opt.timestep

            records.append([data.time, data.qpos[x_id], data.qpos[angle_id], force])

            if show_viewer:
                viewer.sync()
                if data.time >= next_print:
                    print(f"t={data.time:5.1f}s | x={data.qpos[x_id]:+.3f}m | "
                          f"angle={np.degrees(data.qpos[angle_id]):+.2f}deg | F={force:+.3f}N")
                    next_print += 2.0
                delay = model.opt.timestep - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)

        if show_viewer and viewer.is_running():
            print("Simulation finished. Close the viewer to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

    finally:
        if viewer is not None:
            viewer.close()

    return np.array(records)


# Simulate both controllers under identical physical conditions.
reference_result = simulate("reference")
integral_result = simulate("integral", show_viewer=True)

if len(reference_result) and len(integral_result):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for result, label in [(reference_result, "Reference compensation"),
                          (integral_result, "Integral control")]:
        t, x, angle, force = result.T
        axes[0].plot(t, x, label=label)
        axes[1].plot(t, np.degrees(angle), label=label)
        axes[2].plot(t, force, label=label)

    axes[0].axhline(TARGET_POSITION, color="black", linestyle="--", label="Target")
    for ax in axes:
        ax.axvline(DISTURBANCE_START, color="gray", linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Cart position [m]")
    axes[1].set_ylabel("Pole angle [deg]")
    axes[2].set_ylabel("Control force [N]")
    axes[2].set_xlabel("Time [s]")
    fig.tight_layout()
    output = OUTPUT_DIR / "04_integral_comparison.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print("Saved:", output)
