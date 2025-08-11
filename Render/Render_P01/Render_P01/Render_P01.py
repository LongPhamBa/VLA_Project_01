import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Load model & data
model = mujoco.MjModel.from_xml_path("arena_P01.xml")
data = mujoco.MjData(model)

# Tạo scene và context để render
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

# Kích thước ảnh 84×84
cam_width, cam_height = 84, 84
rgb_buffer = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)

# Chọn camera (theo ID trong XML)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")

# Step mô phỏng
mujoco.mj_step(model, data)

# Cập nhật scene
mujoco.mjv_updateScene(model, data, mujoco.MjvOption(),
                       None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

# Render vào buffer
mujoco.mjr_render(mujoco.MjrRect(0, 0, cam_width, cam_height), scene, context)

# Lấy ảnh RGB
mujoco.mjr_readPixels(rgb_buffer, None,
                      mujoco.MjrRect(0, 0, cam_width, cam_height), context)

# Lật ảnh (MuJoCo trả về bị ngược trục dọc)
rgb_buffer = np.flipud(rgb_buffer)

# Hiển thị
plt.imshow(rgb_buffer)
plt.axis("off")
plt.show()
