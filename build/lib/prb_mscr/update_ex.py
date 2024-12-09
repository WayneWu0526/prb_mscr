import numpy as np
from scipy.optimize import minimize
from prb_mscr.energy_and_grad import prb_energy_and_grad

def prb_update_ex(pr, B, q0=None):
    """
    Update the configuration `q` by minimizing the energy.

    Parameters:
    - pr: dict, structure containing parameters
    - B: numpy array, external magnetic field
    - q0: numpy array, initial guess for `q`, default is zeros

    Returns:
    - q: numpy array, optimized configuration
    - fval: float, minimized energy value
    """
    # Initialize q0 if not provided
    if q0 is None:
        q0 = np.zeros(3 * (pr['N'] - 1))

    # Objective function and gradient
    def objective(q):
        E, grad = prb_energy_and_grad(q, pr, B)
        return E, grad

    # Optimization options
    options = {
        'disp': False,           # Suppress output
        'gtol': 1e-8,            # Gradient tolerance
        'tol': 1e-8             # Function value tolerance
    }

    # Perform optimization
    result = minimize(
        fun=lambda q: objective(q),  # Objective function
        x0=q0,                      # Initial guess
        jac=True,                   # Specify that gradient is provided
        method='BFGS',              # BFGS optimization (similar to fminunc)
        options=options             # Optimization options
    )

    # Extract results
    q = result.x
    fval = result.fun
    return q, fval
