import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Cấu hình
DATASET_DIR = "D:\VLA_Project_01\Collect_Data\Collect_Data\collected_data_80steps"  # Thay đổi thành đường dẫn thực tế
SAMPLE_FILES = 5  # Số file sẽ kiểm tra chi tiết
LABEL_DESC = {
    0: "Dừng",
    1: "Tiến thẳng",
    2: "Tiến + rẽ phải",
    3: "Tiến + rẽ trái",
    4: "Rẽ phải tại chỗ",
    5: "Rẽ trái tại chỗ"
}

def check_single_file(filepath):
    """Kiểm tra chi tiết một file npz"""
    try:
        data = np.load(filepath, allow_pickle=True)
        
        # 1. Kiểm tra cấu trúc file
        print("\n" + "="*50)
        print(f"Kiểm tra file: {os.path.basename(filepath)}")
        print("Các keys có trong file:", data.files)
        
        # 2. Kiểm tra shape dữ liệu
        print("\nShape quan trọng:")
        print("Ảnh:", data['steps/observation/image_front'].shape)
        print("Action:", data['steps/action'].shape)
        print("State:", data['steps/observation/state'].shape)
        print("Action labels:", np.unique(data['steps/action_label']))
        
        # 3. Kiểm tra giá trị
        print("\nGiá trị min/max:")
        print("Action:", np.min(data['steps/action'], axis=0), np.max(data['steps/action'], axis=0))
        print("State:", np.min(data['steps/observation/state'], axis=0), np.max(data['steps/observation/state'], axis=0))
        
        # 4. Kiểm tra metadata
        print("\nMetadata:")
        print("Target:", data['episode_metadata/target_xy'])
        print("Success:", data['episode_metadata/success'])
        print("Số samples:", len(data['steps/observation/image_front']))
        
        # 5. Kiểm tra tính nhất quán
        assert len(data['steps/observation/image_front']) == len(data['steps/action'])
        assert len(data['steps/action']) == len(data['steps/observation/state'])
        print("\n✓ Kiểm tra tính nhất quán: PASSED")
        
        # 6. Hiển thị ảnh mẫu
        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            img = data['steps/observation/image_front'][i]
            plt.imshow(img)
            plt.title(f"Step {i}\nAction: {LABEL_DESC[data['steps/action_label'][i]]}")
        plt.tight_layout()
        plt.show()
        
        # 7. Phân bố action
        plt.figure(figsize=(8, 4))
        plt.hist(data['steps/action_label'], bins=len(LABEL_DESC), rwidth=0.8)
        plt.xticks(list(LABEL_DESC.keys()), list(LABEL_DESC.values()), rotation=45)
        plt.title("Phân bố Action trong Episode")
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra file {filepath}: {str(e)}")
        return False

def check_dataset_stats(dataset_dir):
    """Kiểm tra thống kê toàn bộ dataset"""
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    print(f"\nTổng số file trong dataset: {len(files)}")
    
    # Thống kê tổng quan
    total_samples = 0
    action_counts = np.zeros(len(LABEL_DESC), dtype=int)
    success_count = 0
    
    for fname in tqdm(files, desc="Đang phân tích dataset"):
        data = np.load(os.path.join(dataset_dir, fname), allow_pickle=True)
        total_samples += len(data['steps/observation/image_front'])
        action_counts += np.bincount(data['steps/action_label'], minlength=len(LABEL_DESC))
        if data['episode_metadata/success']:
            success_count += 1
    
    # Hiển thị kết quả
    print("\n" + "="*50)
    print("THỐNG KÊ DATASET")
    print("="*50)
    print(f"Tổng số episodes: {len(files)}")
    print(f"Tổng số samples: {total_samples}")
    print(f"Tỷ lệ episode thành công: {success_count/len(files)*100:.1f}%")
    
    print("\nPHÂN BỐ ACTION TOÀN DATASET:")
    for action_id, count in enumerate(action_counts):
        print(f"{LABEL_DESC[action_id]}: {count} samples ({count/total_samples*100:.1f}%)")
    
    # Vẽ biểu đồ phân bố
    plt.figure(figsize=(10, 5))
    plt.bar(LABEL_DESC.values(), action_counts)
    plt.title("Phân bố Action Toàn Dataset")
    plt.ylabel("Số lượng samples")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Kiểm tra ngẫu nhiên một số file
    all_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.npz')]
    sample_files = np.random.choice(all_files, min(SAMPLE_FILES, len(all_files)), replace=False)
    
    for filepath in sample_files:
        check_single_file(filepath)
    
    # Kiểm tra thống kê toàn dataset
    check_dataset_stats(DATASET_DIR)