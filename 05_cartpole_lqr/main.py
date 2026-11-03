"""Cart-Pole: report pole placement versus LQR (rho=10).

python main.py --controller pole --mode balance
python main.py --controller lqr --mode balance
python main.py --controller lqr --mode step
python main.py --compare --mode step
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles

XML_PATH = Path(__file__).with_name('scene.xml')


def linear_model():
    # Parameters from the student's report.
    M, m, l, I, b, g = 0.2, 0.5, 0.3, 0.3, 0.1, 9.8
    d = I * (M + m) + M * m * l**2
    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m*l*l)*b/d, m*m*l*l*g/d, 0],
        [0, 0, 0, 1],
        [0, -m*l*b/d, (M+m)*m*l*g/d, 0],
    ], dtype=float)
    B = np.array([[0], [(I + m*l*l)/d], [0], [m*l/d]])
    return A, B


def controller_gain(name):
    A, B = linear_model()
    if name == 'pole':
        poles = [-0.23 + 0.2346j, -0.23 - 0.2346j, -2.30, -2.53]
        K = place_poles(A, B, poles).gain_matrix[0]
    else:
        # Report pp. 15-17: rho=10, H=[1,0,0,0], Q=rho*H.T@H, R=1.
        H = np.array([[1., 0., 0., 0.]])
        Q = 10.0 * H.T @ H
        R = np.array([[1.]])
        P = solve_continuous_are(A, B, Q, R) # 연속시간 대수적 리카티 방정식을 푸는 함수
        K = (np.linalg.solve(R, B.T @ P))[0]
    # For equilibrium x = reference, theta = 0, the reference feedforward is K[0].
    return K, K[0]


def run_simulation(controller, mode, duration, viewer_enabled):
    K, Nbar = controller_gain(controller)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    cart = model.joint('cart_joint')
    pole = model.joint('pole_joint')
    x_i, xd_i = cart.qposadr[0], cart.dofadr[0]
    phi_i, phid_i = pole.qposadr[0], pole.dofadr[0]
    motor_id = model.actuator('cart_motor').id
    reference = 1.0 if mode == 'step' else 0.0
    if mode != 'step':
        data.qpos[phi_i] = 0.1  # Start 5.7 degrees from upright.
    mujoco.mj_forward(model, data)
    history = []
    next_print = 0.0
    print_every = 2.0 if mode == 'step' else 0.5

    def step():
        nonlocal next_print
        state = np.array([data.qpos[x_i], data.qvel[xd_i],
                          data.qpos[phi_i], data.qvel[phid_i]])
        force = 0.0 if mode == 'off' else float(-K @ state + Nbar * reference)
        force = float(np.clip(force, -30.0, 30.0))
        data.ctrl[motor_id] = force
        mujoco.mj_step(model, data)
        history.append([data.time, data.qpos[x_i], data.qpos[phi_i], force, reference])
        if viewer_enabled and data.time + 1e-8 >= next_print:
            print(f't={data.time:5.2f}s  cart={data.qpos[x_i]:+7.3f}m  '
                  f'pole={np.rad2deg(data.qpos[phi_i]):+7.2f}deg  force={force:+7.2f}N')
            next_print += print_every

    if not viewer_enabled:
        while data.time < duration - 1e-10:
            step()
        return np.asarray(history)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.3, 0, 0.7]
            viewer.cam.distance = 3.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -12
        saved = False
        while viewer.is_running():
            start = time.monotonic()
            if data.time < duration - 1e-10:
                step()
            elif not saved:
                result = np.asarray(history)
                save_plot([result], [controller], mode)
                print('Simulation ended. Viewer remains open until closed.')
                saved = True
            viewer.sync()
            delay = (model.opt.timestep if not saved else 0.02) - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
        if history and not saved:
            save_plot([np.asarray(history)], [controller], mode)
    return np.asarray(history)


def save_plot(results, names, mode):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for result, name in zip(results, names):
        t, x, phi, force, reference = result.T
        axes[0].plot(t, x, label=name)
        axes[1].plot(t, np.rad2deg(phi), label=name)
        axes[2].plot(t, force, label=name)
    axes[0].plot(results[0][:, 0], results[0][:, 4], 'k--', label='Reference')
    axes[0].set_ylabel('Cart position [m]')
    axes[1].set_ylabel('Pole angle [deg]')
    axes[2].set_ylabel('Force [N]')
    axes[2].set_xlabel('Simulation time [s]')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'Cart-Pole: {mode} / {" vs ".join(names)}')
    fig.tight_layout()
    output = XML_PATH.with_name('result_' + mode + '_' + '_'.join(names) + '.png')
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print('Saved:', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['balance', 'step', 'off'], default='balance')
    parser.add_argument('--controller', choices=['pole', 'lqr'], default='lqr')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--compare', action='store_true', help='Run pole and LQR without viewer and overlay results')
    parser.add_argument('--no-viewer', action='store_true')
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (20.0 if args.mode == 'step' else 15.0)
    if duration <= 0:
        parser.error('--duration must be positive')

    if args.compare:
        names = ['pole', 'lqr']
        results = [run_simulation(name, args.mode, duration, False) for name in names]
        save_plot(results, names, args.mode)
    else:
        K, _ = controller_gain(args.controller)
        print(f'{args.controller} gain K:', np.round(K, 4))
        result = run_simulation(args.controller, args.mode, duration, not args.no_viewer)
        if args.no_viewer:
            save_plot([result], [args.controller], args.mode)


if __name__ == '__main__':
    main()
