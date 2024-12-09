# prb_magnetic_robotics/utils.py

import numpy as np

def skew(vector):
    """
    Compute the skew-symmetric matrix for a 3D vector.

    Parameters:
    - vector: numpy array, shape (3,)

    Returns:
    - skew_matrix: numpy array, shape (3, 3)
    """
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def blkdiag(*arrays):
    """Construct a block diagonal matrix."""
    shapes = np.array([a.shape for a in arrays if a is not None])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrays[0].dtype)
    r, c = 0, 0
    for a in arrays:
        if a is not None:
            out[r:r+a.shape[0], c:c+a.shape[1]] = a
            r += a.shape[0]
            c += a.shape[1]
    return out