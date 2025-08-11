
import mujoco
import numpy as np
import cv2

# === Load model và data ===
model = mujoco.MjModel.from_xml_path("car_with_arena_P01.XML")
data = mujoco.MjData(model)

# === Khởi tạo renderer (84x84 cho mô hình) ===
renderer = mujoco.Renderer(model, height=84, width=84)

# Bước mô phỏng ban đầu
mujoco.mj_forward(model, data)

# === Thiết lập cửa sổ hiển thị ===
cv2.namedWindow("MuJoCo Simulation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MuJoCo Simulation", 420, 420)

# === Vòng lặp mô phỏng và hiển thị ===
while True:
    # Bước mô phỏng vật lý
    mujoco.mj_step(model, data)

    # Cập nhật renderer từ camera "topdown"
    renderer.update_scene(data, camera="topdown")
    rgb_img = renderer.render()

    # Chuyển sang BGR để hiển thị bằng OpenCV
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Phóng to ảnh để dễ xem
    display_img = cv2.resize(bgr_img, (420, 420), interpolation=cv2.INTER_NEAREST)

    # Hiển thị ảnh
    cv2.imshow("MuJoCo Simulation", display_img)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()