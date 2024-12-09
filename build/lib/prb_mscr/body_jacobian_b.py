import numpy as np
from prb_mscr.elastic_potential_energy import prb_elastic_potential_energy
from prb_mscr.magnetic_potential_energy import prb_magnetic_potential_energy

def prb_body_jacobian_b(q, pr, B):
    """
    Compute the body Jacobian with respect to the magnetic field (B).

    Parameters:
    - q: numpy array, shape (3*(N-1),), joint variables
    - pr: dict, contains parameters for elastic and magnetic energy
    - B: numpy array, shape (3*num_magnets,), external magnetic field

    Returns:
    - J: numpy array, Jacobian matrix
    """
    # Compute the Hessian of the elastic potential energy
    _, _, HessEe = prb_elastic_potential_energy(q, pr)

    # Compute the Hessian of the magnetic potential energy and M matrix
    _, _, HessEm, Mmatrix = prb_magnetic_potential_energy(q, pr, B)

    # Combine the Hessians
    S = HessEe + HessEm

    # Compute the body Jacobian
    # Regularization can be added if needed (uncomment below):
    # J = np.linalg.solve(S + 1e-3 * np.eye(3 * (pr['N'] - 1)), Mmatrix)
    # Alternative using pseudo-inverse:
    J = np.linalg.pinv(S) @ Mmatrix

    return J
