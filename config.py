import os
import numpy as np

# Cấu hình chung
XML_PATH = "car_with_arena_P01.XML"
OUTPUT_DIR = "collected_data_80steps"
VISUALIZE = False
NUM_EPISODES = 50
IMAGE_SHAPE = (84, 84, 3)

# Ánh xạ lệnh -> tọa độ mục tiêu
TARGETS = {
    "go to the red pillar": np.array([-3.0, 2.0]),
    "go to the green pillar": np.array([1.0, 4.0]),
    "go to the blue pillar": np.array([3.0, -4.0])
}
COMMANDS = list(TARGETS.keys())

# Tạo thư mục đầu ra
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Cấu hình controller 
CONTROLLER_CONFIG = {
    "k_rho": 1.2,
    "k_alpha": 3.0,
    "vmax": 2.0,
    "wmax": 2.0,
    "dv_max": 0.2,
    "dw_max": 0.2,
    "alpha_fwd": 15,  # độ
    "alpha_snap": 3,  # độ
    "stop_r": 0.21    # bán kính dừng (m)
}
