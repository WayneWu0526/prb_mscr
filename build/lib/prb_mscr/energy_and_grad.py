from prb_mscr.elastic_potential_energy import prb_elastic_potential_energy
from prb_mscr.magnetic_potential_energy import prb_magnetic_potential_energy

def prb_energy_and_grad(q, pr, B):
    """
    Compute the total energy (elastic + magnetic) and its gradient.

    Parameters:
    - q: numpy array, shape (3*(N-1),), joint variables
    - pr: dict, contains parameters for elastic and magnetic energy
    - B: numpy array, shape (3*num_magnets,), external magnetic field

    Returns:
    - E: float, total energy
    - grad: numpy array, gradient of the total energy
    """
    # Compute elastic potential energy and its gradient
    Ee, GradEe, _ = prb_elastic_potential_energy(q, pr)

    # Compute magnetic potential energy and its gradient
    Em, GradEm, _, _ = prb_magnetic_potential_energy(q, pr, B)

    # Total energy
    E = Ee + Em

    # Gradient of total energy
    grad = GradEe + GradEm

    return E, grad
