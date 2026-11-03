"""MuJoCo Cart-Pole: reproduce the report's pole-placement controller.

Examples:
    python main.py --mode off
    python main.py --mode balance
    python main.py --mode step --duration 20
    python main.py --mode balance --no-viewer

off : 제어x
balance : 기울어진 역진자 세우기, 목표 각도/위치 = 0
step : 균형을 유지하며 카트 이동, 목표 위치 = 1, 목표 각도 = 0
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from scipy.signal import place_poles


XML_PATH = Path(__file__).with_name("scene.xml")

# Report parameters (unperturbed plant).
M = 0.2       # Cart mass (kg)
m = 0.5       # Pole mass (kg)
l = 0.3       # Pivot to pole COM (m)
I = 0.3       # Pole inertia about COM (kg*m^2)
b = 0.1       # Cart viscous damping (N*s/m)
g = 9.8       # Gravity (m/s^2)

def report_model():
    """Linearization about the upright equilibrium; [x, xdot, phi, phidot]."""
    den = I * (M + m) + M * m * l**2
    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m*l**2)*b/den, m**2*l**2*g/den, 0],
        [0, 0, 0, 1],
        [0, -m*l*b/den, (M+m)*m*l*g/den, 0],
    ], dtype=float)
    B = np.array([0, (I + m*l**2)/den, 0, m*l/den], dtype=float)
    return A, B


def save_plot(samples, mode):
    arr = np.asarray(samples)
    t, x, xdot, phi, phidot, force, ref = arr.T
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, x, label="Cart x [m]")
    axes[0].plot(t, ref, linestyle="--", label="Reference [m]")
    axes[0].set_ylabel("Cart position [m]")
    axes[0].legend(loc="best")
    axes[1].plot(t, np.rad2deg(phi), label="Pole angle [deg]")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Pole angle [deg]")
    axes[1].legend(loc="best")
    axes[2].plot(t, force, label="Applied force [N]")
    axes[2].set_xlabel("Simulation time [s]")
    axes[2].set_ylabel("Force [N]")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Cart-Pole report reproduction: {mode}")
    fig.tight_layout()
    filename = XML_PATH.with_name(f"result_{mode}.png")
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"Plot saved: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["balance", "step", "off"], default="balance")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    duration = (20.0 if args.mode == "step" else 10.0) if args.duration is None else args.duration
    if duration <= 0:
        parser.error("--duration must be positive")

    A, B = report_model()
    poles = [-0.23 + 0.2346j, -0.23 - 0.2346j, -2.30, -2.53]
    K = place_poles(A, B[:, None], poles).gain_matrix[0]
    print("Calculated pole-placement gain K:", np.round(K, 4))
    print("Report gain K:                 [-0.0936 -0.5765 18.9724 8.8193]")
    # For cart-position reference tracking, Nbar = first component of K.
    # u = -K @ state + Nbar * reference
    Nbar = K[0]

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    cart = model.joint("cart_joint")
    pole = model.joint("pole_joint")
    motor_id = model.actuator("cart_motor").id
    x_i, xdot_i = cart.qposadr[0], cart.dofadr[0]
    phi_i, phidot_i = pole.qposadr[0], pole.dofadr[0]

    # Mode 1: small angular perturbation, stabilize at x=0.
    # Mode 2: x-reference 1 m step (the report's step-response experiment).
    # Mode 3: same perturbation with no control.
    if args.mode in ["balance", "off"]:
        data.qpos[phi_i] = 0.1
    reference = 1.0 if args.mode == "step" else 0.0
    mujoco.mj_forward(model, data)

    samples = []
    next_print = 0.0
    print_interval = 2.0 if args.mode == "step" else 0.5

    def advance_one_step():
        nonlocal next_print
        state = np.array([
            data.qpos[x_i], data.qvel[xdot_i],
            data.qpos[phi_i], data.qvel[phidot_i],
        ], dtype=float)
        if args.mode == "off":
            force = 0.0
        else:
            force = float(-K @ state + Nbar * reference)
        force = float(np.clip(force, -30.0, 30.0))
        data.ctrl[motor_id] = force
        mujoco.mj_step(model, data)
        samples.append([
            data.time, data.qpos[x_i], data.qvel[xdot_i],
            data.qpos[phi_i], data.qvel[phidot_i], force, reference,
        ])
        if data.time + 1e-8 >= next_print:
            print(
                f"t={data.time:5.2f} s | x={data.qpos[x_i]:+7.3f} m | "
                f"pole={np.rad2deg(data.qpos[phi_i]):+7.2f} deg | F={force:+7.3f} N"
            )
            next_print += print_interval

    if args.no_viewer:
        while data.time < duration:
            advance_one_step()
        save_plot(samples, args.mode)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.3, 0.0, 0.7]
            viewer.cam.distance = 3.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -12

        graph_saved = False
        while viewer.is_running():
            start = time.monotonic()
            if data.time < duration:
                advance_one_step()
            elif not graph_saved:
                save_plot(samples, args.mode)
                print("Simulation finished; viewer remains open. Close viewer to exit.")
                graph_saved = True

            viewer.sync()
            elapsed = time.monotonic() - start
            delay = (model.opt.timestep if data.time < duration else 0.02) - elapsed
            if delay > 0:
                time.sleep(delay)

        # Save partial results if the viewer was closed before duration elapsed.
        if samples and not graph_saved:
            save_plot(samples, args.mode)


if __name__ == "__main__":
    main()
