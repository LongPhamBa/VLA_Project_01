import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = "car_with_arena_P01.XML"  # file của bạn

# ==== Helper: quaternion → yaw ====
def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ==== Bộ điều khiển phi-holonom ====
class NonholonomicController:
    def __init__(self,
                 k_rho=1.2, k_alpha=3.0,
                 vmax=2.0, wmax=2.0,
                 dv_max=0.2, dw_max=0.2,
                 alpha_fwd=np.deg2rad(15),
                 alpha_snap=np.deg2rad(3),
                 stop_r=0.21):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.vmax = vmax
        self.wmax = wmax
        self.dv_max = dv_max
        self.dw_max = dw_max
        self.alpha_fwd = alpha_fwd
        self.alpha_snap = alpha_snap
        self.stop_r = stop_r
        self.prev_v = 0.0
        self.prev_w = 0.0

    def step(self, state, goal):
        x, y, yaw = state
        gx, gy = goal

        dx, dy = gx - x, gy - y
        rho = np.hypot(dx, dy)
        theta_g = np.arctan2(dy, dx)
        alpha = wrap_pi(theta_g - yaw)

        # Điều khiển tuân cơ học
        if rho > self.stop_r:
            v_cmd = self.k_rho * rho * max(np.cos(alpha), 0.0)
            if np.cos(alpha) <= 0.0 or abs(alpha) > self.alpha_fwd:
                v_cmd = 0.0
            w_cmd = self.k_alpha * alpha
        else:
            v_cmd, w_cmd = 0.0, 0.0

        # Anti-overshoot
        if abs(alpha) < self.alpha_snap:
            w_cmd = 0.0

        # Giới hạn actuator
        v_cmd = np.clip(v_cmd, 0.0, self.vmax)
        w_cmd = np.clip(w_cmd, -self.wmax, self.wmax)

        # Giới hạn tốc độ thay đổi
        dv = np.clip(v_cmd - self.prev_v, -self.dv_max, self.dv_max)
        dw = np.clip(w_cmd - self.prev_w, -self.dw_max, self.dw_max)
        v = self.prev_v + dv
        w = self.prev_w + dw

        self.prev_v, self.prev_w = v, w
        return v, w, dict(rho=rho, alpha=alpha)

# ==== Lấy trạng thái xe từ freejoint ====
def get_car_state(model, data, body_name="car"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    qposadr = model.jnt_qposadr[model.body_jntadr[bid]]
    x, y, z = data.qpos[qposadr:qposadr+3]
    quat = data.qpos[qposadr+3:qposadr+7]
    yaw = quat_to_yaw(quat)
    return np.array([x, y, yaw])

# ==== Main ====
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Lấy ID actuator
    forward_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forward")
    turn_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "turn")

    ctrl = NonholonomicController(
        k_rho=1.0, k_alpha=4.0,
        vmax=2.0, wmax=2.0,
        dv_max=0.2, dw_max=0.2,
        alpha_fwd=np.deg2rad(12),
        alpha_snap=np.deg2rad(2.5),
    )

    # Mục tiêu: ví dụ cột đỏ
    goal_xy = np.array([-3.0, 2.0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_t = time.time()
        while viewer.is_running():
            x, y, yaw = get_car_state(model, data)
            v, w, dbg = ctrl.step((x, y, yaw), goal_xy)

            data.ctrl[forward_id] = float(v)
            data.ctrl[turn_id] = float(w)

            mujoco.mj_step(model, data)

            print(f"rho={dbg['rho']:.3f}, alpha(deg)={np.degrees(dbg['alpha']):.2f}, v={v:.2f}, w={w:.2f}")

            viewer.sync()
            now = time.time()
            if now - last_t < 1/240:
                time.sleep(max(0.0, 1/240 - (now-last_t)))
            last_t = now

if __name__ == "__main__":
    main()
