import numpy as np
from prb_mscr.utils import skew, blkdiag
from prb_mscr.rotation_jacobian_right import prb_rotation_jacobian_right

def prb_magnetic_potential_energy(q, pr, b):
    """
    Compute magnetic potential energy, its gradient, Hessian, and M matrix.

    Parameters:
    - q: numpy array, shape (3*(N-1),), joint variables
    - pr: dict, contains:
        - 'N': int, number of segments
        - 'ind': list of magnet indices
        - 'm': numpy array, magnetic moments of each magnet (3 x num_magnets)
    - b: numpy array, shape (3*num_magnets,), external magnetic field

    Returns:
    - Em: float, magnetic potential energy
    - GradEm: numpy array, gradient of the magnetic potential energy
    - HessEm: numpy array, Hessian of the magnetic potential energy
    - M: numpy array, matrix describing the effect of the field
    """
    theta = q.reshape(3, pr["N"] - 1, order='F')
    Em = 0
    GradEm = np.zeros((3, pr["N"] - 1))
    HessEm = np.zeros((3 * (pr["N"] - 1), 3 * (pr["N"] - 1)))
    M = np.zeros((3 * (pr["N"] - 1), 3 * len(pr["ind"])))

    for i, k in enumerate(pr["ind"]):
        R = prb_rotation_matrix(theta, 0, k) # 0 to k
        if pr["m"].ndim == 1:
            Em -= (R @ pr["m"]).T @ b[3 * i:3 * (i + 1)]
            A = skew(R @ pr["m"])
        else:
            Em -= (R @ pr["m"][:, i]).T @ b[3 * i:3 * (i + 1)]
            A = skew(R @ pr["m"][:, i])

        
        B = skew(b[3 * i:3 * (i + 1)])

        T = [np.zeros((3, 3)) for _ in range(pr["N"] - 1)]
        Q = [np.zeros((3, 3)) for _ in range(pr["N"] - 1)]

        for j in range(k + 1):
            Rj = prb_rotation_matrix(theta, 0, j) # 0 to j
            Jr = prb_rotation_jacobian_right(theta[:, j])
            GradEm[:, j] += (A @ Rj @ Jr).T @ b[3 * i:3 * (i + 1)]
            M[3 * j:3 * (j + 1), 3 * i:3 * (i + 1)] = (A.T @ Rj @ Jr).T
            T[j] = Rj @ Jr
            Q[j] = prb_q(theta[:, j], Rj.T @ A.T @ b[3 * i:3 * (i + 1)])

        D = blkdiag(*T)
        K = blkdiag(2 * np.triu(np.ones((k, k))) - np.eye(k), np.zeros((pr["N"] - k - 1, pr["N"] - k - 1)))
        C = np.kron(K, B.T @ A)
        HessEm += D.T @ prb_symm(C) @ D + prb_symm(blkdiag(*Q))

    GradEm = GradEm.flatten(order='F')
    return Em, GradEm, HessEm, M

from modern_robotics import MatrixExp3, VecToso3

def prb_rotation_matrix(theta, st, ed):
    """
    Compute the rotation matrix for a segment.

    Parameters:
    - theta: numpy array, shape (3, N-1), joint variables
    - st: int, start index
    - ed: int, end index

    Returns:
    - R: numpy array, shape (3, 3), rotation matrix
    """
    R = np.eye(3)
    for i in range(ed, st - 1, -1):
        R = MatrixExp3(VecToso3(theta[:, i])) @ R
    return R

def prb_q(theta, rho):
    """
    Compute Q matrix.

    Parameters:
    - theta: numpy array, shape (3,), joint variable for one segment
    - rho: numpy array, shape (3,), auxiliary vector

    Returns:
    - fval: numpy array, shape (3, 3), Q matrix
    """
    t = np.linalg.norm(theta) + 1e-8
    a = (t - np.sin(t)) / t**3
    b = (1 - np.cos(t)) / t**2
    c = (1 - np.cos(t) - 0.5 * t**2) / t**4
    d = 3 * (t - np.sin(t) - t**3 / 6) / t**5

    rhox = skew(rho)
    thetax = skew(theta)
    fval = (2 * a - b) * (thetax @ rhox) - \
           (a + 2 * c) * (thetax @ thetax @ rhox) - \
           (c - d) * (thetax @ rhox @ thetax @ thetax)
    return fval

def prb_symm(A):
    """
    Compute the symmetric part of a matrix.

    Parameters:
    - A: numpy array, matrix to symmetrize

    Returns:
    - fval: numpy array, symmetric matrix
    """
    return 0.5 * (A + A.T)
