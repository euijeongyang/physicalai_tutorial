import time
from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = Path(__file__).with_name("scene.xml")

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(str(XML_PATH)) # 모델 불러오기
data = mujoco.MjData(model) # 시뮬레이션 데이터 생성 -> 시뮬레이션 중 변하는 상태 저장됨
'''
data.qpos 관절 위치 및 자세
data.qvel 관절 속도
data.ctrl Actuator 제어 입력
data.time 현재 시뮬레이션 시간
'''

# Run simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    ball_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ball"
    )

    while viewer.is_running():

        step_start = time.time()

        # Physics simulation
        # 호출될 때마다 지정된 시간 간격만큼 물리 시뮬레이션 진행
        if data.time < 3.0:
            mujoco.mj_step(model, data)

        ball_height = data.xpos[ball_id][2]

        print(
            f"Time: {data.time:.3f} s, "
            f"Height: {ball_height:.3f} m"
        )

        # Update viewer
        viewer.sync()

        # Real-time simulation
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)