"""08: Robustness test based on report pages 30-32.

Run: python 08_robustness.py
Change DISTURBANCE and PARAMETER_ERROR below to compare conditions.
Output: output/08_robustness_*.png
"""

import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles

from common import XML_PATH, OUTPUT_DIR, linear_model


# Experiment settings
DISTURBANCE = "sine"      # "step", "sine", or "noise"
PARAMETER_ERROR = 0.10    # 0.0, 0.10, or -0.10

DURATION = 20.0
DISTURBANCE_START = 5.0
TARGET_POSITION = 1.0
DISTURBANCE_FORCE = 0.05  # N; sine amplitude / noise standard deviation

SHOW_VIEWER = False

# Design all controllers using the nominal (unmodified) report model.
A, B = linear_model()
C = np.array([[1.0, 0.0, 0.0, 0.0]])
poles = [-0.23 + 0.2346j, -0.23 - 0.2346j, -2.30, -2.53]

K_pole = place_poles(A, B, poles).gain_matrix[0]
P = solve_continuous_are(A, B, 10.0 * C.T @ C, np.array([[1.0]]))
K_lqr = (B.T @ P)[0]

# Integral controller: z_dot = cart_position - target_position
A_int = np.block([
    [A, np.zeros((4, 1))],
    [C, np.zeros((1, 1))]
])
B_int = np.vstack([B, [[0.0]]])
K_int = place_poles(A_int, B_int, poles + [-2.0]).gain_matrix[0]
Kx, Ki = K_int[:4], K_int[4]


def simulate(controller, viewer_enabled=False):
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    cart = model.joint("cart_joint")
    pole = model.joint("pole_joint")
    x_id, vx_id = cart.qposadr[0], cart.dofadr[0]
    angle_id, omega_id = pole.qposadr[0], pole.dofadr[0]
    motor_id = model.actuator("cart_motor").id

    # Change the physical plant, NOT the controller design model.
    # Mass, inertia and viscous damping are scaled together.
    model.body_mass[1:] *= 1.0 + PARAMETER_ERROR
    model.body_inertia[1:] *= 1.0 + PARAMETER_ERROR
    model.dof_damping[:] *= 1.0 + PARAMETER_ERROR
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(42)  # Same disturbance sequence for every controller
    integral_error = 0.0
    next_print = 0.0
    records = []

    viewer = None
    if viewer_enabled:
        viewer = mujoco.viewer.launch_passive(model, data)
        with viewer.lock():
            viewer.cam.lookat[:] = [0.3, 0, 0.7]
            viewer.cam.distance = 3.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -12

    try:
        while data.time < DURATION - 1e-9:
            if viewer is not None and not viewer.is_running():
                break
            start = time.monotonic()

            x = data.qpos[x_id]
            state = np.array([x, data.qvel[vx_id],
                              data.qpos[angle_id], data.qvel[omega_id]])

            if controller == "pole":
                command = -K_pole @ state + K_pole[0] * TARGET_POSITION
            elif controller == "lqr":
                command = -K_lqr @ state + K_lqr[0] * TARGET_POSITION
            else:
                command = -Kx @ state - Ki * integral_error

            force = float(np.clip(command, -30.0, 30.0))
            data.ctrl[motor_id] = force

            # External disturbance is independent of the control command.
            disturbance = 0.0
            if data.time >= DISTURBANCE_START:
                if DISTURBANCE == "step":
                    disturbance = DISTURBANCE_FORCE
                elif DISTURBANCE == "sine":
                    disturbance = DISTURBANCE_FORCE * np.sin(2.0 * data.time)
                elif DISTURBANCE == "noise":
                    disturbance = rng.normal(0.0, DISTURBANCE_FORCE)

            data.qfrc_applied[vx_id] = disturbance
            mujoco.mj_step(model, data)

            if controller == "integral" and -30.0 < command < 30.0:
                integral_error += (x - TARGET_POSITION) * model.opt.timestep

            records.append([data.time, data.qpos[x_id],
                            np.degrees(data.qpos[angle_id]), force, disturbance])

            if viewer is not None:
                viewer.sync()
                if data.time >= next_print - 1e-9:
                    print(f"t={data.time:5.1f}s | x={data.qpos[x_id]:+.3f}m | "
                          f"angle={np.degrees(data.qpos[angle_id]):+.2f}deg | "
                          f"F={force:+.3f}N")
                    next_print += 2.0
                delay = model.opt.timestep - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)

        if viewer is not None and viewer.is_running():
            print("Simulation finished. Close the viewer to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    finally:
        if viewer is not None:
            viewer.close()

    return np.array(records)


if DISTURBANCE not in ("step", "sine", "noise"):
    raise ValueError("DISTURBANCE must be 'step', 'sine', or 'noise'")
if PARAMETER_ERROR <= -1.0:
    raise ValueError("PARAMETER_ERROR must be greater than -1")

results = {}
for controller in ("pole", "lqr", "integral"):
    results[controller] = simulate(controller)

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for controller, label in (("pole", "Pole placement"),
                          ("lqr", "LQR"),
                          ("integral", "Integral")):
    values = results[controller]
    t, x, angle, force, disturbance = values.T
    axes[0].plot(t, x, label=label)
    axes[1].plot(t, angle, label=label)
    axes[2].plot(t, force, label=label)

axes[0].axhline(TARGET_POSITION, color="black", linestyle="--", label="Target")
for ax in axes:
    ax.axvline(DISTURBANCE_START, color="gray", linestyle=":")
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Cart position [m]")
axes[1].set_ylabel("Pole angle [deg]")
axes[2].set_ylabel("Control force [N]")
axes[2].set_xlabel("Time [s]")
fig.suptitle(f"Disturbance: {DISTURBANCE} | Parameter error: {PARAMETER_ERROR:+.0%}")
fig.tight_layout()
output = OUTPUT_DIR / f"05_robustness_{DISTURBANCE}_{PARAMETER_ERROR:+.0%}.png"
fig.savefig(output, dpi=140)
plt.close(fig)
print("Saved:", output)

# Finite-horizon approximation (not an infinite-time steady-state guarantee).
print("\nFinal 2-second average position error:")
for controller, values in results.items():
    last = values[values[:, 0] >= DURATION - 2.0]
    if len(last):
        error = np.mean(np.abs(last[:, 1] - TARGET_POSITION))
        print(f"{controller:>8}: {error:.4f} m")

if SHOW_VIEWER:
    print("\nMuJoCo viewer: integral controller under the same condition")
    simulate("integral", viewer_enabled=True)
