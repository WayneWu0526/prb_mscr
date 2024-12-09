import numpy as np

def prb_voronoi_length(L):
    """
    Calculate the Voronoi lengths of segments.

    Parameters:
    - L: numpy array, lengths of the segments

    Returns:
    - ell: numpy array, Voronoi lengths
    """
    return (L[:-1] + L[1:]) / 2


def prb_elastic_potential_energy(q, pr):
    """
    Calculate the elastic potential energy, its gradient, and Hessian.

    Parameters:
    - q: numpy array, shape (3*(N-1),), joint variables
    - pr: dict, containing:
        - 'L': numpy array, segment lengths
        - 'G': numpy array, shear modulus for each segment
        - 'Ixx': numpy array, second moment of area for each segment
        - 'E': numpy array, Young's modulus for each segment
        - 'I': numpy array, second moment of area (I) for bending

    Returns:
    - Ee: float, elastic potential energy
    - GradEe: numpy array, gradient of elastic potential energy
    - HessEe: numpy array, Hessian of elastic potential energy
    """
    # Voronoi lengths
    ell = prb_voronoi_length(pr['L'])

    # 计算克罗内克积的三个分量
    kron1 = np.kron(pr["G"] * pr["Ixx"] / ell, [1, 0, 0])
    kron2 = np.kron(pr["E"] * pr["I"] / ell, [0, 1, 0])
    kron3 = np.kron(pr["E"] * pr["I"] / ell, [0, 0, 1])

    # 求和得到对角矩阵的对角线
    Lambda_diagonal = kron1 + kron2 + kron3

    # 转换为对角矩阵
    Lambda = np.diag(Lambda_diagonal)

    # Elastic potential energy
    Ee = 0.5 * np.dot(q.T, np.dot(Lambda, q))

    # Gradient of elastic potential energy
    GradEe = np.dot(Lambda, q)

    # Hessian of elastic potential energy
    HessEe = Lambda

    return Ee, GradEe, HessEe
