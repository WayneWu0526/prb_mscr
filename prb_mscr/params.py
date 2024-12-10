# params.py

import numpy as np
import math

# 柔性段长度 (m)
L_flex = 18e-3
# 刚性段长度 (m)
L_rigid = 2e-3
# 导管总长度 (m)
L = L_flex + L_rigid
# 导管半径 (m)
r = 1e-3
# 横截面积 (m^2)
A = math.pi * r**2
# 惯性矩 (m^4)
I = math.pi * r**4 / 4
# 杨氏模量 (Pa)
E = 5e6
# 泊松比
nu = 0.49
# 剪切模量 (Pa)
G = E / (2 * (nu + 1))
# 磁偶极矩模量 (A·m^2)
m_norm = 1e-2
# 磁偶极矩向量
m = np.array([m_norm, 0, 0])
# 外部磁场模量 (T)
B_norm = 30e-2
# 外部磁场向量
B = B_norm * np.array([0, 0, 1])
