# theta.py
import numpy as np
from modern_robotics import RpToTrans

def prb_theta(q, pr):
    """
    PRB Theta Calculation.

    Parameters:
    - q: numpy array, joint variables
    - pr: dict, structure containing `N` (number of segments) and `L` (segment lengths)

    Returns:
    - S: numpy array, screw axes (6 x (N - 1))
    - M: numpy array, transformation matrix of the end-effector (4x4)
    - P: numpy array, positions of nodes (3 x N)
    """
    N = pr['N']
    L = pr['L']

    M = RpToTrans(np.eye(3), np.array([sum(L), 0, 0]))  # End-effector
    S = np.zeros((6, N - 1))
    P = np.zeros((3, N))

    k = 0
    for i in range(0, 3 * (N - 1), 3):
        P[:, k] = [sum(L[:k+1]), 0, 0]
        S[:, k] = np.concatenate((q[i:i+3], -np.cross(q[i:i+3], P[:, k])))
        k += 1

    P[:, N - 1] = [sum(L), 0, 0]
    return S, M, P
