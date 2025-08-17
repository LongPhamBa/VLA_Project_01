import mujoco
import mujoco.viewer
import numpy as np
import time

from controller import NonholonomicController, get_action_label
from randomizer import randomize_visuals  # ✅ thêm để test luôn random visuals
import config  # XML_PATH, CONTROLLER_CONFIG, TARGETS

# ==== Helper: quaternion → yaw ====
def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

# ==== Reset random vị trí + hướng + visuals ====
def reset_random_pose(model, data, apply_randomization=True):
    # Kích thước an toàn trong sân
    arena_limit = 5.5  # tường ở ±6.1 m
    min_distance_from_targets = 0.5  # m

    # Lấy vị trí các mục tiêu từ config
    target_positions = np.array(list(config.TARGETS.values()))

    # Random vị trí hợp lệ (không gần mục tiêu)
    while True:
        start_pos = np.random.uniform(-arena_limit, arena_limit, size=2)
        if np.all(np.linalg.norm(target_positions - start_pos, axis=1) > min_distance_from_targets):
            break

    # Random góc ban đầu
    start_yaw = np.random.uniform(-np.pi, np.pi)

    z = 0.03
    qw = np.cos(start_yaw / 2.0)
    qz = np.sin(start_yaw / 2.0)

    qpos0 = np.copy(data.qpos)
    qpos0[0:3] = np.array([start_pos[0], start_pos[1], z])
    qpos0[3:7] = np.array([qw, 0.0, 0.0, qz])
    data.qpos[:] = qpos0

    # Random visuals nếu cần
    if apply_randomization:
        randomize_visuals(model)

    mujoco.mj_forward(model, data)


# ==== Lấy trạng thái xe từ freejoint ====
def get_car_state(model, data, body_name="car"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    qposadr = model.jnt_qposadr[model.body_jntadr[bid]]
    x, y, _ = data.qpos[qposadr:qposadr+3]
    quat = data.qpos[qposadr+3:qposadr+7]
    yaw = quat_to_yaw(quat)
    return np.array([x, y, yaw])

# ==== Bảng mô tả nhãn hành động ====
LABEL_DESC = {
    0: "Dừng",
    1: "Tiến thẳng",
    2: "Tiến + rẽ phải",
    3: "Tiến + rẽ trái",
    4: "Rẽ phải đứng tại chỗ",
    5: "Rẽ trái đứng tại chỗ"
}

def main():
    # Load model
    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)

    # Lấy ID actuator
    forward_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forward")
    turn_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "turn")

    # Khởi tạo controller
    ctrl = NonholonomicController(
        k_rho=config.CONTROLLER_CONFIG["k_rho"],
        k_alpha=config.CONTROLLER_CONFIG["k_alpha"],
        vmax=config.CONTROLLER_CONFIG["vmax"],
        wmax=config.CONTROLLER_CONFIG["wmax"],
        dv_max=config.CONTROLLER_CONFIG["dv_max"],
        dw_max=config.CONTROLLER_CONFIG["dw_max"],
        alpha_fwd=np.deg2rad(config.CONTROLLER_CONFIG["alpha_fwd"]),
        alpha_snap=np.deg2rad(config.CONTROLLER_CONFIG["alpha_snap"]),
        stop_r=config.CONTROLLER_CONFIG["stop_r"]
    )

    # Chọn mục tiêu ngẫu nhiên
    command = np.random.choice(config.COMMANDS)
    goal_xy = config.TARGETS[command]
    print(f"Target: {command} -> {goal_xy}")

    # Reset random pose + visuals
    reset_random_pose(model, data, apply_randomization=True)

    step_count = 0
    logic_step_count = 0
    prev_label = None
    same_count = 0
    THRESHOLD = 15

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_t = time.time()
        while viewer.is_running():
            x, y, yaw = get_car_state(model, data)
            v, w = ctrl.step((x, y, yaw), goal_xy)

            data.ctrl[forward_id] = float(v)
            data.ctrl[turn_id] = float(w)

            label = get_action_label(v, w)
            label_desc = LABEL_DESC.get(label, "Không xác định")

            mujoco.mj_step(model, data)
            step_count += 1

            # Kiểm tra logic step
            if label == prev_label:
                same_count += 1
                if same_count >= THRESHOLD:
                    logic_step_count += 1
                    same_count = 0
                    print(f"LogicStep {logic_step_count:03d} | Step {step_count:03d} "
                          f"| x={x:.2f}, y={y:.2f}, yaw={np.degrees(yaw):.1f}°, "
                          f"v={v:.2f}, w={w:.2f}, label={label} ({label_desc})")
            else:
                same_count = 1
                logic_step_count += 1
                print(f"LogicStep {logic_step_count:03d} | Step {step_count:03d} "
                      f"| x={x:.2f}, y={y:.2f}, yaw={np.degrees(yaw):.1f}°, "
                      f"v={v:.2f}, w={w:.2f}, label={label} ({label_desc})")

            prev_label = label

            # Kiểm tra tới mục tiêu
            dist_to_goal = np.linalg.norm(np.array([x, y]) - goal_xy)
            if dist_to_goal < 0.21:
                print(f"✅ Đã tới mục tiêu sau {step_count} step thực tế "
                      f"({logic_step_count} step logic), dừng.")
                break

            viewer.sync()
            now = time.time()
            if now - last_t < 1/240:
                time.sleep(max(0.0, 1/240 - (now - last_t)))
            last_t = now

if __name__ == "__main__":
    main()
