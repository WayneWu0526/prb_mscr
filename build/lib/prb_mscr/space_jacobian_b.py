import numpy as np
from prb_mscr.elastic_potential_energy import prb_elastic_potential_energy
from prb_mscr.magnetic_potential_energy import prb_magnetic_potential_energy
from prb_mscr.space_jacobian import prb_space_jacobian

def prb_space_jacobian_b(q, pr, B):
    """
    Compute the Jacobian matrix with respect to the magnetic field (B).

    Parameters:
    - q: numpy array, shape (3*(N-1),), joint variables
    - pr: dict, contains parameters for elastic and magnetic energy
    - B: numpy array, shape (3*num_magnets,), external magnetic field

    Returns:
    - Jbs: numpy array, Jacobian matrix with respect to the magnetic field
    """
    # Compute the Hessian of the elastic potential energy
    _, _, HessEe = prb_elastic_potential_energy(q, pr)

    # Compute the Hessian of the magnetic potential energy and M matrix
    _, _, HessEm, Mmatrix = prb_magnetic_potential_energy(q, pr, B)

    # Combine the Hessians
    S = HessEe + HessEm

    # Compute the space Jacobian
    Jtheta = prb_space_jacobian(q, pr)

    # Compute the Jacobian with respect to the magnetic field
    Jbs = Jtheta @ np.linalg.solve(S, Mmatrix)

    return Jbs


