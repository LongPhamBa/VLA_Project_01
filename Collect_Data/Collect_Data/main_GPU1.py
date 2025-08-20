import os
from multiprocessing import Process, Queue, cpu_count
from simulation import DataCollectorLogic
from config import OUTPUT_DIR, NUM_EPISODES

# Ép dùng Nvidia GPU (qua NVIDIA Control Panel đã set High performance)
# Backend OpenGL phù hợp trên Windows
os.environ["MUJOCO_GL"] = "glfw"

NUM_WORKERS = min(4, cpu_count())   # tối ưu cho i5-1240P + MX570 (2GB VRAM)


def build_task_queue(num_episodes):
    """Xây dựng hàng đợi các episode chưa có"""
    q = Queue()

    existing = {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    }

    for idx in range(num_episodes):
        if idx not in existing:
            q.put(idx)

    return q


def worker_loop(q: Queue):
    """Worker lấy job từ queue và thu thập dữ liệu"""
    collector = DataCollectorLogic()
    backend = os.environ.get("MUJOCO_GL", "default")
    print(f"[Worker {os.getpid()}] Running with backend={backend}")

    while not q.empty():
        try:
            episode_idx = q.get_nowait()
        except Exception:
            break

        print(f"[Worker {os.getpid()}] Collecting episode {episode_idx:05d}")
        collector.collect_dataset(start_idx=episode_idx, num_episodes=1)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_queue = build_task_queue(NUM_EPISODES)
    total_tasks = task_queue.qsize()

    if total_tasks == 0:
        print("✅ All episodes already collected.")
        exit(0)

    print(f"🚀 Starting collection with {NUM_WORKERS} workers "
          f"→ {total_tasks} missing episodes will be generated.")

    processes = []
    for _ in range(NUM_WORKERS):
        p = Process(target=worker_loop, args=(task_queue,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All workers finished, dataset is complete.")
