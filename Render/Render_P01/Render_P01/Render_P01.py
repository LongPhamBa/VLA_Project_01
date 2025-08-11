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

# Render từ camera "topdown"
renderer.update_scene(data, camera="topdown")
rgb_image = renderer.render()   # Ảnh RGB 84x84 cho mô hình

# ================== [SHOW ẢNH - CÓ THỂ XÓA SAU] ==================
# Convert sang BGR để OpenCV hiển thị
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Cho phép cửa sổ thay đổi kích thước
cv2.namedWindow("Topdown Camera View", cv2.WINDOW_NORMAL)

# Phóng to ảnh để dễ xem (vd: 420x420)
display_image = cv2.resize(bgr_image, (420, 420), interpolation=cv2.INTER_NEAREST)

# (Tùy chọn) Đặt kích thước cửa sổ lớn hơn
cv2.resizeWindow("Topdown Camera View", 420, 420)

# Hiển thị ảnh
cv2.imshow("Topdown Camera View", display_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
# ===============================================================