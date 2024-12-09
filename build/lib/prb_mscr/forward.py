import numpy as np
from modern_robotics import MatrixExp6, VecTose3, RpToTrans, TransToRp, MatrixLog6, se3ToVec
from prb_mscr.theta import prb_theta

def prb_forward(q, pr):
    """
    PRB Forward Kinematics.

    Parameters:
    - q: numpy array, joint variables
    - pr: dict, structure containing `N` (number of segments) and `L` (segment lengths)

    Returns:
    - T: numpy array, final transformation matrix (4x4)
    - Pose: numpy array, pose of each node (6 x N)
    - Qp: numpy array, quaternion and position of each node (7 x N)
    """
    N = pr['N']

    Pose = np.zeros((6, N))  # 6 x N matrix for pose
    Qp = np.zeros((7, N))   # 7 x N matrix for quaternion and position

    # Call PRB_Theta function
    S, M, P = prb_theta(q, pr)

    T = np.eye(4)  # Initial transformation matrix (Identity)
    for i in range(N):
        if i < N - 1:
            T = T @ MatrixExp6(VecTose3(S[:, i]))  # Update T with exponential map
            M0 = RpToTrans(np.eye(3), P[:, i])    # Transformation to current node
            T_ = T @ M0
        else:
            T_ = T @ M  # For the end-effector

        Pose[:, i] = se3ToVec(MatrixLog6(T_))  # Log map to get pose vector
        R, p = TransToRp(T_)  # Decompose transformation into R and p
        quat = rotm2quat(R)   # Convert rotation matrix to quaternion
        Qp[:, i] = np.concatenate((quat, p))  # Combine quaternion and position

    return T, Pose, Qp


def rotm2quat(R):
    """
    Converts a rotation matrix to a quaternion.

    Parameters:
    - R: 3x3 numpy array, rotation matrix

    Returns:
    - quat: 4-element numpy array, quaternion [qw, qx, qy, qz]
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return np.array([qw, qx, qy, qz])
