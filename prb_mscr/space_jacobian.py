import numpy as np
from modern_robotics import MatrixExp6, VecTose3, Adjoint
from prb_mscr.theta import prb_theta
from prb_mscr.utils import skew
from prb_mscr.rotation_jacobian_right import prb_rotation_jacobian_right

def prb_space_jacobian(q, pr):
    """
    Compute the space Jacobian for the pseudo-rigid-body model.

    Parameters:
    - q: numpy array, shape (3*(N-1),), joint variables
    - pr: dict, structure containing `N` (number of segments) and other parameters

    Returns:
    - Js: numpy array, shape (6, 3*(N-1)), space Jacobian matrix
    """
    S, _, P = prb_theta(q, pr)
    ExpXi = np.eye(4)  # Initial transformation matrix
    Js = np.zeros((6, 3 * (pr['N'] - 1)))  # Space Jacobian

    for i in range(pr['N'] - 1):
        Theta = S[0:3, i]
        Rho = S[3:, i]

        # Compute Pose Jacobian Left
        JHl = prb_pose_jacobian_left(Rho, Theta)

        # Compute partial derivative of Xi with respect to Theta
        parXiparTheta = np.vstack([np.eye(3), skew(P[:, i])])

        # Update the space Jacobian
        Js[:, 3 * i:3 * (i + 1)] = Adjoint(ExpXi) @ JHl @ parXiparTheta

        # Update ExpXi for the next segment
        ExpXi = ExpXi @ MatrixExp6(VecTose3(S[:, i]))

    return Js

def prb_pose_jacobian_left(Rho, Theta):
    """
    Compute the Pose Jacobian Left.

    Parameters:
    - Rho: numpy array, shape (3,), translation vector
    - Theta: numpy array, shape (3,), rotation vector

    Returns:
    - Jl: numpy array, shape (6, 6), pose Jacobian left
    """
    t = np.linalg.norm(Theta) + 1e-8
    rhox = skew(Rho)
    thetax = skew(Theta)

    # Compute Q matrix
    Q = (0.5 * rhox +
         (t - np.sin(t)) / t**3 * (thetax @ rhox + rhox @ thetax + thetax @ rhox @ thetax) -
         (1 - t**2 / 2 - np.cos(t)) / t**4 *
         (thetax @ thetax @ rhox + rhox @ thetax @ thetax - 3 * thetax @ rhox @ thetax) -
         0.5 * ((1 - t**2 / 2 - np.cos(t)) / t**4 -
                3 * (t - np.sin(t) - t**3 / 6) / t**5) *
         (thetax @ rhox @ thetax @ thetax + thetax @ thetax @ rhox @ thetax))

    # Compute Rotation Jacobian Right (transpose)
    jl = prb_rotation_jacobian_right(Theta).T

    # Combine into the full Pose Jacobian Left
    Jl = np.block([[jl, np.zeros((3, 3))], [Q, jl]])
    return Jl
