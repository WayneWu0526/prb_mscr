import numpy as np

def set_param_function(L, E, r, nu, B_norm, L_rigid, Nm, N, m):
    """
    Initialize the parameter dictionary for the pseudo-rigid-body model.

    Parameters:
    - L: list or numpy array, lengths of the segments
    - E: float, Young's modulus (Pa)
    - r: float, radius of the catheter (m)
    - nu: float, Poisson's ratio
    - B_norm: float, external magnetic field strength (T)
    - L_rigid: float, rigid segment length threshold (m)
    - Nm: int, number of magnetic segments
    - N: int, number of flexible segments per segment group
    - m: numpy array, magnetic moment vector (3 x num_magnets)

    Returns:
    - pr: dict, parameter dictionary
    """
    # Initialize parameter dictionary
    pr = {}

    pr["L"] = [0]
    if L[0] == 0:
        L = L[1:]

    pr["ind"] = []
    for i in range(0, len(L) - 1, 2):
        if (L[i] / N) > L_rigid:
            pr["L"].extend([L[i] / N] * N)
            pr["L"].extend([L[i + 1] / Nm] * Nm)
        else:
            pr["L"].append(L[i])
            pr["L"].extend([L[i + 1] / Nm] * Nm)
        pr["ind"].append(len(pr["L"]) - 2)  # 修复索引逻辑

    pr["L"] = np.array(pr["L"])

    pr["N"] = len(pr["L"])
    pr["Nm"] = len(pr["ind"])
    pr["E"] = np.full(pr["N"] - 1, E)
    pr["r"] = np.full(pr["N"] - 1, r)
    pr["A"] = np.pi * pr["r"] ** 2
    pr["I"] = pr["A"] * pr["r"] ** 2 / 4
    pr["Ixx"] = 2 * pr["I"]
    pr["E"][pr["ind"]] = 1e8  # 刚性部分的模量
    pr["nu"] = nu
    pr["G"] = pr["E"] / (2 * (pr["nu"] + 1))
    pr["B_norm"] = B_norm
    pr["mu0"] = 4 * np.pi * 10e-7
    pr["k"] = 342.86
    pr["m"] = m

    return pr