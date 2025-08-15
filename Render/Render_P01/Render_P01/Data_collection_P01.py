import os
import time
import json
import numpy as np
import mujoco
import cv2

# -----------------------------
# BLOCK 0: Cấu hình chung
# -----------------------------
XML_PATH = "car_with_arena_P01.XML"
OUTPUT_DIR = "collected_data_80steps"
VISUALIZE = False
NUM_EPISODES = 5
IMAGE_SHAPE = (84, 84, 3)

# Ánh xạ lệnh -> tọa độ mục tiêu
TARGETS = {
    "đi tới bệ màu đỏ": np.array([-3.0, 2.0]),
    "đi tới bệ màu xanh lá": np.array([1.0, 4.0]),
    "đi tới bệ màu xanh dương": np.array([3.0, -4.0])
}
COMMANDS = list(TARGETS.keys())

os.makedirs(OUTPUT_DIR, exist_ok=True)

existing_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("episode_") and f.endswith(".npz")])
start_episode_idx = int(existing_files[-1].split("_")[1].split(".")[0]) + 1 if existing_files else 0
print(f"Bắt đầu thu thập từ tập {start_episode_idx:05d}")

# -----------------------------
# BLOCK 1: Tải mô hình
# -----------------------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1])

CAR_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
ACT_FORWARD_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forward")
ACT_TURN_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "turn")

# Lấy yaw chuẩn từ quaternion
def get_car_xy():
    return data.xpos[CAR_BODY_ID][:2].copy()

def get_car_yaw():
    qposadr = model.jnt_qposadr[model.body_jntadr[CAR_BODY_ID]]
    quat = data.qpos[qposadr+3:qposadr+7]
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

# -----------------------------
# BLOCK 2: Bộ điều khiển phi-holonom (bản đã chạy tốt)
# -----------------------------
class NonholonomicController:
    def __init__(self,
                 k_rho=1.0, k_alpha=4.0,
                 vmax=2.0, wmax=2.0,
                 dv_max=0.2, dw_max=0.2,
                 alpha_fwd=np.deg2rad(12),
                 alpha_snap=np.deg2rad(2.5)):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.vmax = vmax
        self.wmax = wmax
        self.dv_max = dv_max
        self.dw_max = dw_max
        self.alpha_fwd = alpha_fwd
        self.alpha_snap = alpha_snap
        self.prev_v = 0.0
        self.prev_w = 0.0

    def reset(self):
        self.prev_v = 0.0
        self.prev_w = 0.0

    def step(self, current_pos, current_yaw, target_pos):
        dx, dy = target_pos - current_pos
        rho = np.hypot(dx, dy)
        target_angle = np.arctan2(dy, dx)
        alpha = (target_angle - current_yaw + np.pi) % (2 * np.pi) - np.pi

        v_cmd = self.k_rho * rho * max(np.cos(alpha), 0.0)
        if np.cos(alpha) <= 0.0 or abs(alpha) > self.alpha_fwd:
            v_cmd = 0.0
        w_cmd = self.k_alpha * alpha

        if abs(alpha) < self.alpha_snap:
            w_cmd = 0.0

        v_cmd = np.clip(v_cmd, 0.0, self.vmax)
        w_cmd = np.clip(w_cmd, -self.wmax, self.wmax)

        dv = np.clip(v_cmd - self.prev_v, -self.dv_max, self.dv_max)
        dw = np.clip(w_cmd - self.prev_w, -self.dw_max, self.dw_max)
        v = self.prev_v + dv
        w = self.prev_w + dw

        self.prev_v, self.prev_w = v, w
        return float(v), float(w)

controller = NonholonomicController()

def oracle_control(current_pos, current_yaw, target_pos):
    return controller.step(current_pos, current_yaw, target_pos)

def get_action_label(forward_cmd, turn_cmd):
    if abs(forward_cmd) < 1e-3 and abs(turn_cmd) < 1e-3:
        return 3
    elif abs(turn_cmd) > 0.1:
        return 2 if turn_cmd > 0 else 1
    elif abs(forward_cmd) > 0.1:
        return 0

# -----------------------------
# BLOCK 3: Ngẫu nhiên hóa
# -----------------------------
def randomize_visuals():
    for gi in range(model.ngeom):
        rgba = model.geom_rgba[gi].copy()
        jitter = (np.random.rand(4) - 0.5) * 0.3
        rgba[:3] = np.clip(rgba[:3] + jitter[:3], 0.05, 1.0)
        rgba[3] = 1.0
        model.geom_rgba[gi] = rgba
    for li in range(model.nlight):
        diff = model.light_diffuse[li].copy()
        diff = np.clip(diff + (np.random.rand(3) - 0.5) * 0.6, 0.1, 2.0)
        model.light_diffuse[li] = diff

# -----------------------------
# BLOCK 4: Reset tập
# -----------------------------
def reset_episode(start_pos=None, start_yaw=None, apply_randomization=True):
    mujoco.mj_resetData(model, data)
    controller.reset()  # reset vận tốc điều khiển mỗi tập
    if apply_randomization:
        randomize_visuals()
    if start_pos is None:
        start_pos = np.random.uniform(-1.0, 1.0, size=2)
    if start_yaw is None:
        start_yaw = np.random.uniform(-np.pi, np.pi)
    if model.nq >= 7:
        z = 0.03
        qw = np.cos(start_yaw / 2.0)
        qz = np.sin(start_yaw / 2.0)
        qpos0 = np.zeros(model.nq)
        qpos0[0:3] = np.array([start_pos[0], start_pos[1], z])
        qpos0[3:7] = np.array([qw, 0.0, 0.0, qz])
        data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)

# -----------------------------
# BLOCK 5: Thu thập dữ liệu
# -----------------------------
def collect_dataset(start_idx, num_episodes=NUM_EPISODES, visualize=VISUALIZE):
    meta_path = os.path.join(OUTPUT_DIR, "metadata_80.json")
    all_meta = json.load(open(meta_path, "r", encoding="utf-8")) if os.path.exists(meta_path) else []
    episode_idx, saved_count = start_idx, 0

    if visualize:
        cv2.namedWindow("MuJoCo Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MuJoCo Simulation", 420, 420)

    try:
        while episode_idx < num_episodes:
            command = np.random.choice(COMMANDS)
            target_pos = TARGETS[command]
            reset_episode(apply_randomization=True)
            episode_data = []
            max_steps = np.random.randint(30, 81)

            for step in range(max_steps):
                car_xy, car_yaw = get_car_xy(), get_car_yaw()
                forward_cmd, turn_cmd = oracle_control(car_xy, car_yaw, target_pos)
                label = int(get_action_label(forward_cmd, turn_cmd))
                data.ctrl[ACT_FORWARD_ID] = forward_cmd
                data.ctrl[ACT_TURN_ID] = turn_cmd
                mujoco.mj_step(model, data)
                renderer.update_scene(data, camera="topdown")
                rgb_img = renderer.render()

                episode_data.append({
                    "image": rgb_img.astype(np.uint8),
                    "command": command,
                    "action_continuous": [forward_cmd, turn_cmd],
                    "action_label": label
                })

                if np.linalg.norm(car_xy - target_pos) < 0.25:
                    break
                if visualize:
                    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("MuJoCo Simulation", cv2.resize(bgr, (420, 420)))
                    if cv2.waitKey(1) & 0xFF == 27:
                        episode_idx = num_episodes
                        break

            fname = os.path.join(OUTPUT_DIR, f"episode_{episode_idx:05d}.npz")
            images = np.stack([x["image"] for x in episode_data])
            commands = np.array([x["command"] for x in episode_data], dtype=object)
            actions_cont = np.stack([x["action_continuous"] for x in episode_data])
            action_labels = np.array([x["action_label"] for x in episode_data], dtype=np.int32)

            np.savez_compressed(fname, images=images, commands=commands,
                                actions_continuous=actions_cont, action_labels=action_labels)

            all_meta.append({"episode": episode_idx, "command": command, "steps": len(episode_data), "file": fname})
            episode_idx += 1
            saved_count += 1

            if saved_count % 50 == 0:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(all_meta, f, indent=2, ensure_ascii=False)
                print(f"Đã lưu {saved_count} tập (mới nhất: {fname})")

    finally:
        all_meta = []
        for fname in sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npz")]):
            path = os.path.join(OUTPUT_DIR, fname)
            try:
                data_npz = np.load(path, allow_pickle=True)
                commands = data_npz["commands"]
                all_meta.append({
                    "episode": int(fname.split("_")[1].split(".")[0]),
                    "command": str(commands[0]) if len(commands) > 0 else "",
                    "steps": len(commands),
                    "file": path
                })
            except Exception as e:
                print(f"Lỗi khi đọc {fname}: {e}")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)
        print(f"Đã cập nhật metadata_80.json với {len(all_meta)} tập")

    if visualize:
        cv2.destroyAllWindows()
    print(f"Hoàn thành thu thập. Đã lưu {saved_count} tập tại {OUTPUT_DIR}")

# -----------------------------
# BLOCK 6: Chạy chính
# -----------------------------
if __name__ == "__main__":
    collect_dataset(start_episode_idx, visualize=True)
