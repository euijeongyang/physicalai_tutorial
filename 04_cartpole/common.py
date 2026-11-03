"""Cart-Pole common model, simulation, and plotting functions."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


XML_PATH = Path(__file__).with_name("scene.xml")
OUTPUT_DIR = Path(__file__).with_name("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def linear_model():
    # Parameters from the control-system-design report
    M, m, l, I, b, g = 0.2, 0.5, 0.3, 0.3, 0.1, 9.8
    d = I * (M + m) + M * m * l**2

    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m*l*l)*b/d, m*m*l*l*g/d, 0],
        [0, 0, 0, 1],
        [0, -m*l*b/d, (M + m)*m*l*g/d, 0],
    ])
    B = np.array([[0], [(I + m*l*l)/d], [0], [m*l/d]])
    return A, B


def run(K, mode="balance", duration=None, L=None, name="result"):
    """Run MuJoCo. L=None: full state feedback; L: estimated state feedback."""
    if mode not in ("off", "balance", "step"):
        raise ValueError("mode must be 'off', 'balance', or 'step'")
    if duration is None:
        duration = 20.0 if mode == "step" else 5.0

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    cart = model.joint("cart_joint")
    pole = model.joint("pole_joint")
    x_id, vx_id = cart.qposadr[0], cart.dofadr[0]
    a_id, va_id = pole.qposadr[0], pole.dofadr[0]
    motor_id = model.actuator("cart_motor").id

    if mode != "step":
        data.qpos[a_id] = 0.1  # Initial pole tilt, radians
    mujoco.mj_forward(model, data)

    A, B = linear_model()
    C = np.array([1.0, 0.0, 0.0, 0.0])  # Measure cart position only
    estimate = np.zeros(4)
    reference = 1.0 if mode == "step" else 0.0
    Nbar = K[0]  # Constant-position reference compensation
    history = []
    next_print = 0.0
    print_interval = 2.0 if mode == "step" else 0.5
    finished = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.3, 0, 0.7]
            viewer.cam.distance = 3.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -12

        while viewer.is_running():
            start = time.monotonic()
            if data.time < duration - 1e-9:
                state = np.array([
                    data.qpos[x_id], data.qvel[vx_id],
                    data.qpos[a_id], data.qvel[va_id]
                ])

                feedback_state = state if L is None else estimate
                force = -K @ feedback_state + Nbar * reference
                if mode == "off":
                    force = 0.0
                force = float(np.clip(force, -30, 30))
                data.ctrl[motor_id] = force

                # Full-order observer: xhat_dot = A xhat + B u + L(y - C xhat)
                if L is not None:
                    measurement = state[0]
                    error = measurement - C @ estimate
                    estimate += model.opt.timestep * (A @ estimate + B[:, 0] * force + L * error)

                mujoco.mj_step(model, data)
                history.append([
                    data.time, data.qpos[x_id], data.qpos[a_id],
                    force, reference, estimate[0], estimate[2]
                ])

                if data.time >= next_print - 1e-9:
                    print(f"t={data.time:5.2f}s | x={data.qpos[x_id]:+6.3f}m | "
                          f"angle={np.degrees(data.qpos[a_id]):+6.2f}deg | F={force:+6.2f}N")
                    next_print += print_interval

            elif not finished:
                plot(history, name, mode, L is not None)
                print("Simulation finished. Close the viewer to exit.")
                finished = True

            viewer.sync()
            delay = (0.02 if finished else model.opt.timestep) - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)

    if history and not finished:
        plot(history, name, mode, L is not None)


def plot(history, name, mode, observer):
    values = np.array(history)
    t, x, angle, force, reference, x_hat, angle_hat = values.T

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, x, label="Cart position")
    axes[0].plot(t, reference, "k--", label="Target")
    axes[1].plot(t, np.degrees(angle), label="Pole angle")
    axes[2].plot(t, force, label="Control force")

    if observer:
        axes[0].plot(t, x_hat, ":", label="Estimated position")
        axes[1].plot(t, np.degrees(angle_hat), ":", label="Estimated angle")

    axes[0].set_ylabel("Position [m]")
    axes[1].set_ylabel("Angle [deg]")
    axes[2].set_ylabel("Force [N]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output = OUTPUT_DIR / f"{name}_{mode}.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print("Saved:", output)
