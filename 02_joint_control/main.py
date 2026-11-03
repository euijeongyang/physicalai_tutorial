import time
import math
from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = Path(__file__).with_name("scene.xml")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

joint_id = model.joint("joint1").id # 관절 정보 가져오기
actuator_id = model.actuator("motor1").id

# joint1 관절의 id 가져오기
qpos_id = model.jnt_qposadr[joint_id]
qvel_id = model.jnt_dofadr[joint_id]

with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.cam.lookat[:] = [0, 0, 1.4]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

    while viewer.is_running() and data.time < 10.0:

        step_start = time.time()

        # Target joint angle, 목표 관절 각도 설정
        target_angle = 0.5 * math.sin(data.time)
        # target_angle = 1.0 * math.sin(2.0 * data.time)

        # Position control
        data.ctrl[actuator_id] = target_angle

        # Physics simulation
        mujoco.mj_step(model, data)

        # Joint state
        joint_angle = data.qpos[qpos_id]
        joint_velocity = data.qvel[qvel_id]

        print(
            f"Time: {data.time:.3f} | "
            f"Target: {target_angle:.3f} | "
            f"Angle: {joint_angle:.3f} | "
            f"Velocity: {joint_velocity:.3f}"
        )

        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)