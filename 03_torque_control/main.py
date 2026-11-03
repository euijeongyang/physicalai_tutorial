import time
import math
from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = Path(__file__).with_name("scene.xml")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

joint_id = model.joint("joint1").id
actuator_id = model.actuator("motor1").id

qpos_id = model.jnt_qposadr[joint_id]
qvel_id = model.jnt_dofadr[joint_id]

# PD gains
kp = 200.0
kd = 20.0

# Simulation settings
simulation_time = 5.0
print_interval = 0.5
next_print_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Camera settings
    with viewer.lock():
        viewer.cam.lookat[:] = [0, 0, 1.4]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

    while viewer.is_running():

        step_start = time.time()

        if data.time < simulation_time:

            # Target trajectory
            target_angle = 0.5 * math.sin(2.0 * data.time)
            target_velocity = 1.0 * math.cos(2.0 * data.time)

            # Joint states
            joint_angle = data.qpos[qpos_id]
            joint_velocity = data.qvel[qvel_id]

            # PD controller
            position_error = target_angle - joint_angle
            velocity_error = target_velocity - joint_velocity

            torque = kp * position_error + kd * velocity_error

            # Torque limit
            torque = max(-100.0, min(100.0, torque))

            # Torque control, 관절에 torque[Nm]만큼 토크 명령
            data.ctrl[actuator_id] = torque

            # Physics simulation
            mujoco.mj_step(model, data)

            # Print every 0.5 seconds
            if data.time >= next_print_time:

                print(
                    f"Time: {data.time:.2f} | "
                    f"Target: {target_angle:.3f} | "
                    f"Angle: {data.qpos[qpos_id]:.3f} | "
                    f"Torque: {torque:.3f}"
                )

                next_print_time += print_interval

        # Update viewer
        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)