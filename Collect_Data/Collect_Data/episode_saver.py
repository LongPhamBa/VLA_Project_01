import numpy as np
import os

def save_episode(fname, episode_data, target_pos, success_flag, seed):
    images = np.stack([x["image"] for x in episode_data]).astype(np.uint8)
    commands = np.array([x["command"] for x in episode_data], dtype=object)
    actions_raw = np.stack([x["action_continuous"] for x in episode_data]).astype(np.float32)
    action_labels = np.array([x["action_label"] for x in episode_data], dtype=np.int32)
    states = np.stack([x["state"] for x in episode_data]).astype(np.float32)
    times_arr = np.array([x["time"] for x in episode_data], dtype=np.float32)

    a_min = np.array([-2.0, -2.0], dtype=np.float32)
    a_max = np.array([ 2.0,  2.0], dtype=np.float32)
    actions_norm = np.clip((actions_raw - a_min) / (a_max - a_min) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    n = len(episode_data)
    is_first = np.zeros((n,), dtype=bool); is_first[0] = True
    is_last  = np.zeros((n,), dtype=bool); is_last[-1]  = True

    np.savez_compressed(
        fname,
        **{
            "episode_metadata/task_name": np.array("drive_to_pillar", dtype=object),
            "episode_metadata/unnorm_key": np.array("car_forward_turn_v1", dtype=object),
            "episode_metadata/action_space/min": a_min,
            "episode_metadata/action_space/max": a_max,
            "episode_metadata/target_xy": np.array(target_pos, dtype=np.float32),
            "episode_metadata/success": np.array(success_flag, dtype=bool),
            "episode_metadata/random_seed": np.array(seed, dtype=np.int32),
            "episode_metadata/state_units": np.array(["m", "m", "rad"], dtype=object),
            "episode_metadata/action_units": np.array(["m/s", "rad/s"], dtype=object),
            "steps/observation/image_front": images,
            "steps/observation/state": states,
            "steps/language_instruction": commands,
            "steps/action": actions_norm,
            "steps/action_raw": actions_raw,
            "steps/action_label": action_labels,
            "steps/is_first": is_first,
            "steps/is_last": is_last,
            "steps/time": times_arr
        }
    )
