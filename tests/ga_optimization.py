import numpy as np
from prb_mscr import PRBModel
from prb_mscr import params
def main():

    L_rigid = params.L_rigid
    L_flex = 18e-3
    # L = [L_flex, L_rigid, L_flex, L_rigid, L_flex, L_rigid]
    L = [L_flex, L_rigid]
    # m = np.array([[1e-2, 0, 0], [1e-2, 0, 0], [1e-2, 0, 0]]).T
    m = np.array([1e-2, 0, 0]).T
    # b = 1e-2 * np.random.rand(9, 1)
    # b = 30e-2 * np.array([0, 0, 1, 0, 0, 1, 0, 0, 1]).T
    b = 30e-2 * np.array([0, 0, 1]).T
    
    model = PRBModel.initialize(L=L, m=m, Nm=2, N=3)
    model.update_magnetic_field(b)
    
    q, _ = model.update_ex()
    
    T, _, _ = model.forward(q)

    print(T)
    

if __name__ == "__main__":
    main()