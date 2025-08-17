import os
from multiprocessing import Process
from simulation import DataCollectorLogic
from config import OUTPUT_DIR

TOTAL_END = 10000    # kết thúc ở episode 10000
NUM_WORKERS = 4      # số core muốn dùng

def get_start_episode_idx():
    """Tìm episode kế tiếp dựa trên file hiện có"""
    existing_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR)
         if f.startswith("episode_") and f.endswith(".npz")]
    )
    return int(existing_files[-1].split("_")[1].split(".")[0]) + 1 if existing_files else 0

def run_worker(start_idx, end_idx):
    """Worker chạy một đoạn episode"""
    collector = DataCollectorLogic()
    num_eps = end_idx - start_idx + 1
    print(f"[Worker {os.getpid()}] Collecting {num_eps} episodes ({start_idx:05d} → {end_idx:05d})")
    collector.collect_dataset(start_idx=start_idx, num_episodes=num_eps, visualize=False)

if __name__ == "__main__":
    start_idx = get_start_episode_idx()   # vd: 807
    print(f"Starting parallel collection from {start_idx:05d}")

    remaining = TOTAL_END - start_idx + 1
    step = remaining // NUM_WORKERS
    processes = []

    for i in range(NUM_WORKERS):
        s = start_idx + i * step
        e = start_idx + (i + 1) * step - 1 if i < NUM_WORKERS - 1 else TOTAL_END
        if s <= e:
            p = Process(target=run_worker, args=(s, e))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("✅ All workers finished.")
