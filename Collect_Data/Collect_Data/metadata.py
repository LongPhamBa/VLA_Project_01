import os, json
import numpy as np

def rebuild_metadata(output_dir, meta_path):
    all_meta = []
    for fname in sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")]):
        path = os.path.join(output_dir, fname)
        try:
            data_npz = np.load(path, allow_pickle=True)
            commands = data_npz["steps/language_instruction"]
            success = bool(data_npz["episode_metadata/success"])
            all_meta.append({
                "episode": int(fname.split("_")[1].split(".")[0]),
                "command": str(commands[0]) if len(commands) > 0 else "",
                "steps": len(commands),
                "file": path,
                "success": success
            })
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"📄 Updated {meta_path} with {len(all_meta)} episodes")

