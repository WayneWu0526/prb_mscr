# 例如 prb_magnetic_robotics/kinematics.py
import numpy as np
from prb_mscr.utils import skew  # 从 utils 导入 skew 函数

def prb_rotation_jacobian_right(Omg):
    """
    Compute the right Jacobian of SO(3) for a given angular velocity vector.

    Parameters:
    - Omg: numpy array, shape (3,), angular velocity vector

    Returns:
    - Jr: numpy array, shape (3, 3), the right Jacobian matrix
    """
    omg = np.linalg.norm(Omg)  # Magnitude of the angular velocity vector

    if omg == 0:
        return np.eye(3)  # Identity matrix when angular velocity is zero
    else:
        skew_Omg = skew(Omg)  # 调用 utils 中的 skew 函数
        term1 = (1 - np.cos(omg)) / (omg ** 2) * skew_Omg
        term2 = (omg - np.sin(omg)) / (omg ** 3) * np.dot(skew_Omg, skew_Omg)
        return np.eye(3) - term1 + term2
