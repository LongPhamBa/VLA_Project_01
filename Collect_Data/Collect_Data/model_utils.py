import numpy as np
import mujoco

def load_model(xml_path):
    return mujoco.MjModel.from_xml_path(xml_path)

def initialize_renderer(model, image_shape):
    return mujoco.Renderer(model, height=image_shape[0], width=image_shape[1])

def get_body_id(model, body_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

def get_actuator_id(model, actuator_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

def get_car_state(model, data, car_body_id):
    """Lấy trạng thái xe [x, y, yaw] giống Oracle_controller.py"""
    qposadr = model.jnt_qposadr[model.body_jntadr[car_body_id]]
    x, y, _ = data.qpos[qposadr:qposadr+3]
    quat = data.qpos[qposadr+3:qposadr+7]
    yaw = _quat_to_yaw(quat)
    return np.array([x, y, yaw])

def _quat_to_yaw(q):
    """Chuyển quaternion sang yaw (nội bộ)"""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)