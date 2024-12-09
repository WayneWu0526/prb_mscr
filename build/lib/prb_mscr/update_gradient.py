import numpy as np
from prb_mscr.energy_and_grad import prb_energy_and_grad

def prb_gradient_descent(pr, B, q0=None, lr=1e-3, max_iters=1000, tol=1e-6):
    """
    Perform gradient descent to minimize energy E(q).

    Parameters:
    - pr: dict, problem parameters
    - B: numpy array, external magnetic field
    - q0: numpy array, initial guess for q
    - lr: float, learning rate (step size)
    - max_iters: int, maximum number of iterations
    - tol: float, tolerance for convergence (based on gradient norm)

    Returns:
    - q: numpy array, optimized configuration
    - fval: float, minimized energy value
    """
    if q0 is None:
        q0 = np.zeros(3 * (pr["N"] - 1))  # Initialize q0 to zeros

    q = q0.copy()
    for iteration in range(max_iters):
        # Compute energy and gradient
        E, grad = prb_energy_and_grad(q, pr, B)

        # Update q using gradient descent
        q -= lr * grad

        # Check for convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        # Optionally print progress
        if iteration % 100 == 0 or iteration == max_iters - 1:
            print(f"Iteration {iteration + 1}: E = {E:.6f}, Gradient Norm = {grad_norm:.6e}")

    else:
        print("Gradient descent did not converge within the maximum number of iterations.")

    fval = E
    return q, fval
