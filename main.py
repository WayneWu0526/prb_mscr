import numpy as np
from prb_mscr.params import L_flex, L_rigid, r, E, nu, B_norm, m_norm  # 从参数文件导入初始值
from prb_mscr.set_param_function import set_param_function           # 导入参数设置函数
from prb_mscr.update_ex import prb_update_ex                        # 导入优化函数
from prb_mscr.forward import prb_forward                                # 导入前向函数
from prb_mscr.update_gradient import prb_gradient_descent
from prb_mscr.elastic_potential_energy import prb_elastic_potential_energy
from prb_mscr.magnetic_potential_energy import prb_magnetic_potential_energy
from prb_mscr.body_jacobian import prb_body_jacobian

def main():
    # Step 1: 设置段长和磁矩配置
    L = [L_flex, L_rigid, L_flex, L_rigid, L_flex, L_rigid]  # 柔性和刚性段交替
    # L = [L_flex, L_rigid] # 只有一段柔性和一段刚性
    m = np.array([[m_norm, 0, 0], [m_norm, 0, 0], [m_norm, 0, 0]]).T  # 磁矩方向沿 x 轴
    # m = np.array([m_norm, 0, 0]).T 

    # Step 2: 创建 pr 结构体
    pr = set_param_function(L, E, r, nu, B_norm, L_rigid, Nm=2, N=3, m=m)

    # Step 3: 设置外部磁场 B
    # B = B_norm * (2 * np.random.rand(9) - 1)  # 随机磁场分量
    B = B_norm * np.tile(np.array([0, 0, 1]), len(pr["ind"]))  # 沿 z 轴的均匀磁场

    # Step 4: 利用 update_ex 计算 q
    # q0 = np.ones(3 * (pr["N"] - 1)) * 0.01  # 初始化 q0 为0.01
    q0 = np.zeros(3 * (pr["N"] - 1))  # 初始化 q0 为0
    q, fval = prb_update_ex(pr, B, q0)

    T, Pose, Qp = prb_forward(q, pr) # 计算正运动学
    
    _, _, HessEe = prb_elastic_potential_energy(q, pr) # 计算弹性势能的 Hessian 矩阵
    _, _, HessEm, M = prb_magnetic_potential_energy(q, pr, B) # 计算磁势能的 Hessian 矩阵
    # 输出结果
    # print("优化后的配置 q：")
    print(q)
    print(T)
    
    Jb = prb_body_jacobian(q, pr)
    print(Jb)
    # print("\n最小化的能量值 fval：")
    # print(fval)

if __name__ == "__main__":
    main()
