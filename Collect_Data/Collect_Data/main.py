import os
from simulation import DataCollectorLogic  
from config import OUTPUT_DIR

def get_start_episode_idx():
    existing_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) 
         if f.startswith("episode_") and f.endswith(".npz")]
    )
    return int(existing_files[-1].split("_")[1].split(".")[0]) + 1 if existing_files else 0

if __name__ == "__main__":
    start_episode_idx = get_start_episode_idx()
    print(f"Starting data collection from episode {start_episode_idx:05d}")

    collector = DataCollectorLogic()
    collector.collect_dataset(start_idx=start_episode_idx, visualize=False)
