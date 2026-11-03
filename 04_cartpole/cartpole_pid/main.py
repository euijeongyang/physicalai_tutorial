import time
from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = Path(__file__).with_name("scene.xml")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# Joint IDs
cart_joint = model.joint("cart_joint")
pole_joint = model.joint("pole_joint")

cart_qpos = cart_joint.qposadr[0]
cart_qvel = cart_joint.dofadr[0]

pole_qpos = pole_joint.qposadr[0]
pole_qvel = pole_joint.dofadr[0]

motor_id = model.actuator("cart_motor").id

# Initial state
data.qpos[cart_qpos] = 0.0
data.qpos[pole_qpos] = 0.1

mujoco.mj_forward(model, data)

# PD gains
kp_pole = 100.0
kd_pole = 20.0

kp_cart = 2.0
kd_cart = 10.0

# Simulation settings
simulation_time = 5.0
print_interval = 0.5
next_print_time = 0.0

# Controller
control_enabled = True
# control_enabled = False

with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10

    while viewer.is_running():

        step_start = time.time()

        if data.time < simulation_time:

            # States
            x = data.qpos[cart_qpos]
            x_dot = data.qvel[cart_qvel]

            theta = data.qpos[pole_qpos]
            theta_dot = data.qvel[pole_qvel]

            # PD controller
            if control_enabled:

                force_pole = (
                    kp_pole * theta
                    + kd_pole * theta_dot
                )

                force_cart = (
                    -kp_cart * x
                    -kd_cart * x_dot
                )

                force = force_pole + force_cart

            else:
                force = 0.0

            # Force limit
            force = max(-30.0, min(30.0, force))

            # Control input
            data.ctrl[motor_id] = force

            # Physics simulation
            mujoco.mj_step(model, data)

            # Print
            if data.time >= next_print_time:

                print(
                    f"Time: {data.time:.2f} | "
                    f"Cart: {x:.3f} | "
                    f"Pole: {theta:.3f} | "
                    f"Force: {force:.3f}"
                )

                next_print_time += print_interval

        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)