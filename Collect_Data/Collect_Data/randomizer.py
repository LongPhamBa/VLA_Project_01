import numpy as np

def randomize_visuals(model):
    """Ngẫu nhiên hóa màu sắc và ánh sáng trong mô hình"""
    for gi in range(model.ngeom):
        rgba = model.geom_rgba[gi].copy()
        jitter = (np.random.rand(4) - 0.5) * 0.3
        rgba[:3] = np.clip(rgba[:3] + jitter[:3], 0.05, 1.0)
        rgba[3] = 1.0
        model.geom_rgba[gi] = rgba
        
    for li in range(model.nlight):
        diff = model.light_diffuse[li].copy()
        diff = np.clip(diff + (np.random.rand(3) - 0.5) * 0.6, 0.1, 2.0)
        model.light_diffuse[li] = diff