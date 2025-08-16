import os
import time
import cv2
import numpy as np
import mujoco

from config import (
    OUTPUT_DIR, COMMANDS, TARGETS, NUM_EPISODES, VISUALIZE,
    CONTROLLER_CONFIG, XML_PATH
)
from controller import NonholonomicController, get_action_label
from model_utils import get_car_state
from randomizer import randomize_visuals
from episode_saver import save_episode
from metadata import rebuild_metadata


class DataCollectorLogic:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # ID actuator
        self.act_forward_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forward"
        )
        self.act_turn_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "turn"
        )

        # ID body của xe
        self.car_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "car"
        )

        # Controller
        self.controller = NonholonomicController(
            k_rho=CONTROLLER_CONFIG["k_rho"],
            k_alpha=CONTROLLER_CONFIG["k_alpha"],
            vmax=CONTROLLER_CONFIG["vmax"],
            wmax=CONTROLLER_CONFIG["wmax"],
            dv_max=CONTROLLER_CONFIG["dv_max"],
            dw_max=CONTROLLER_CONFIG["dw_max"],
            alpha_fwd=np.deg2rad(CONTROLLER_CONFIG["alpha_fwd"]),
            alpha_snap=np.deg2rad(CONTROLLER_CONFIG["alpha_snap"]),
            stop_r=CONTROLLER_CONFIG["stop_r"]
        )

        # Renderer
        self.renderer = mujoco.Renderer(self.model, 84, 84)

    def reset_random_pose(self, target_pos, arena_limit=5.5, min_dist_target=0.5):
        """Random vị trí và góc xe, tránh gần mục tiêu và tường"""
        while True:
            start_pos = np.random.uniform(-arena_limit, arena_limit, size=2)
            if np.linalg.norm(start_pos - target_pos) > min_dist_target:
                break
        start_yaw = np.random.uniform(-np.pi, np.pi)

        z = 0.03
        qw = np.cos(start_yaw / 2.0)
        qz = np.sin(start_yaw / 2.0)

        qpos0 = np.copy(self.data.qpos)
        qpos0[0:3] = np.array([start_pos[0], start_pos[1], z])
        qpos0[3:7] = np.array([qw, 0.0, 0.0, qz])
        self.data.qpos[:] = qpos0

    def reset_episode(self, target_pos, apply_randomization=True, random_seed=None):
        mujoco.mj_resetData(self.model, self.data)
        if random_seed is not None:
            np.random.seed(random_seed)
        self.reset_random_pose(target_pos)
        if apply_randomization:
            randomize_visuals(self.model)
        mujoco.mj_forward(self.model, self.data)

    def collect_dataset(self, start_idx, num_episodes=NUM_EPISODES, visualize=VISUALIZE):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        meta_path = os.path.join(OUTPUT_DIR, "metadata.json")

        episode_idx = start_idx
        THRESHOLD = 15          # gộp step giống nhau
        MAX_LOGIC_STEPS = 150   # giới hạn số logic step

        if visualize:
            cv2.namedWindow("MuJoCo Simulation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MuJoCo Simulation", 420, 420)

        try:
            while episode_idx < start_idx + num_episodes:
                command = np.random.choice(COMMANDS)
                target_pos = TARGETS[command]
                seed = np.random.randint(0, 1e9)

                self.reset_episode(target_pos, apply_randomization=True, random_seed=seed)

                episode_data = []
                prev_label = None
                same_count = 0
                start_time = time.time()
                success_flag = False
                step_logic = 0   # đếm số logic step

                while True:
                    car_state = get_car_state(self.model, self.data, self.car_body_id)
                    forward_cmd, turn_cmd = self.controller.step(car_state, target_pos)
                    label = int(get_action_label(forward_cmd, turn_cmd))

                    self.data.ctrl[self.act_forward_id] = forward_cmd
                    self.data.ctrl[self.act_turn_id] = turn_cmd
                    mujoco.mj_step(self.model, self.data)

                    self.renderer.update_scene(self.data, camera="topdown")
                    rgb_img = self.renderer.render()

                    # gộp step giống nhau
                    if label == prev_label:
                        same_count += 1
                        if same_count < THRESHOLD:
                            continue
                        else:
                            same_count = 0
                    else:
                        same_count = 1
                        prev_label = label

                    # tăng logic step
                    step_logic += 1

                    episode_data.append({
                        "image": rgb_img.astype(np.uint8),
                        "command": command,
                        "action_continuous": [forward_cmd, turn_cmd],
                        "action_label": label,
                        "state": car_state.astype(np.float32),
                        "time": time.time() - start_time
                    })

                    # điều kiện dừng
                    if np.linalg.norm(car_state[:2] - target_pos) < 0.21:
                        success_flag = True
                        break
                    if step_logic >= MAX_LOGIC_STEPS:
                        print(f"Episode {episode_idx:05d} reached max logic steps ({MAX_LOGIC_STEPS}), stopping.")
                        break

                    if visualize:
                        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("MuJoCo Simulation", cv2.resize(bgr, (420, 420)))
                        if cv2.waitKey(1) & 0xFF == 27:
                            raise KeyboardInterrupt()

                # Lưu dữ liệu episode
                fname = os.path.join(OUTPUT_DIR, f"episode_{episode_idx:05d}.npz")
                save_episode(fname, episode_data, target_pos, success_flag, seed)
                print(f"Saved episode: {fname}")

                episode_idx += 1
        finally:
            rebuild_metadata(OUTPUT_DIR, meta_path)
            if visualize:
                cv2.destroyAllWindows()