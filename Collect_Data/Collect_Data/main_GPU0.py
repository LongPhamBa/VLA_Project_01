import os
from multiprocessing import Process
from simulation import DataCollectorLogic
from config import OUTPUT_DIR, NUM_EPISODES

# ⚡ GPU render qua GLFW (Windows + Nvidia)
os.environ["MUJOCO_GL"] = "glfw"

# Số tiến trình song song (tối ưu cho i5-1240P + MX570)
NUM_WORKERS = 4


def get_start_episode_idx():
    """Tìm episode kế tiếp dựa trên file hiện có"""
    existing_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR)
         if f.startswith("episode_") and f.endswith(".npz")]
    )
    return int(existing_files[-1].split("_")[1].split(".")[0]) + 1 if existing_files else 0


def run_worker(start_idx, num_eps):
    """Worker chạy một nhóm episode"""
    collector = DataCollectorLogic()
    print(f"[Worker {os.getpid()}] Collecting {num_eps} episodes (from {start_idx:05d})")
    collector.collect_dataset(start_idx=start_idx, num_episodes=num_eps)


if __name__ == "__main__":
    start_idx = get_start_episode_idx()
    total_end = start_idx + NUM_EPISODES - 1

    print(f" Starting collection {start_idx:05d} → {total_end:05d} "
          f"using {NUM_WORKERS} workers (GLFW + GPU)")

    step = NUM_EPISODES // NUM_WORKERS
    processes = []

    for i in range(NUM_WORKERS):
        s = start_idx + i * step
        n = step if i < NUM_WORKERS - 1 else NUM_EPISODES - step * (NUM_WORKERS - 1)
        if n > 0:
            p = Process(target=run_worker, args=(s, n))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print(" All workers finished.")
