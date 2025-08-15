import numpy as np

path = "episode_00003.npz"
data = np.load(path, allow_pickle=True)

print("Keys trong file:", list(data.keys()))
for k in data.files:
    print(f"{k} shape:", data[k].shape if hasattr(data[k], 'shape') else "object")


# Kiểm tra vị trí 1-2 bước đầu
if "actions" in data:
    print("First 5 actions:", data["actions"][:65])
if "action_labels" in data:
    print("First 5 labels:", data["action_labels"][:65])
