import numpy as np

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

    def reset(self):
        self.prev_v = 0.0
        self.prev_w = 0.0

    def step(self, state, goal):
        x, y, yaw = state
        gx, gy = goal

        dx, dy = gx - x, gy - y
        rho = np.hypot(dx, dy)
        theta_g = np.arctan2(dy, dx)
        alpha = self._wrap_pi(theta_g - yaw)

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
        return v, w

    def _wrap_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

def get_action_label(forward_cmd, turn_cmd):
    if abs(forward_cmd) < 0.05 and abs(turn_cmd) < 0.05:
        return 0 # Dừng
    elif forward_cmd > 0.05 and abs(turn_cmd) <= 0.05:
        return 1 # Tiến thẳng
    elif forward_cmd > 0.05 and turn_cmd > 0.05:
        return 2 # Tiến + rẽ phải
    elif forward_cmd > 0.05 and turn_cmd < -0.05:
        return 3 # Tiến + rẽ trái
    elif turn_cmd > 0.05:
        return 4 # Rẽ phải đứng tại chỗ
    elif turn_cmd < -0.05:
        return 5 # Rẽ trái đứng tại chỗ
