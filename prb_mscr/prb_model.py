from .config import DEFAULT_CONFIG
from .forward import prb_forward
from .update_ex import prb_update_ex
from .update_gradient import prb_gradient_descent
from .set_param_function import set_param_function
from .elastic_potential_energy import prb_elastic_potential_energy
from .magnetic_potential_energy import prb_magnetic_potential_energy
from .energy_and_grad import prb_energy_and_grad
from .body_jacobian import prb_body_jacobian
from .body_jacobian_b import prb_body_jacobian_b
from .space_jacobian import prb_space_jacobian
from .space_jacobian_b import prb_space_jacobian_b

class PRBModel:
    def __init__(self, pr, B):
        self.pr = pr
        self.B = B

    @classmethod
    def initialize(cls, **kwargs):
        """
        初始化 PRBModel 的参数配置。
        """
        config = DEFAULT_CONFIG.copy()
        config.update(kwargs)
        
        pr = set_param_function(
            L=config["L"], 
            E=config["E"], 
            r=config["r"], 
            nu=config["nu"], 
            B_norm=config["B_norm"], 
            L_rigid=config["L_rigid"], 
            Nm=config["Nm"], 
            N=config["N"], 
            m=config["m"]
        )
        B = [0, 0, config["B_norm"]]
        return cls(pr, B)
    
    def update_magnetic_field(self, B):
        """
        更新磁场配置。
        """
        self.B = B

    def forward(self, q):
        """计算正向运动学"""
        return prb_forward(q, self.pr)
    
    def elastic_potential_energy(self, q):
        """计算弹性势能"""
        return prb_elastic_potential_energy(q, self.pr)
    
    def magnetic_potential_energy(self, q):
        """计算磁势能"""
        return prb_magnetic_potential_energy(q, self.pr, self.B)
    
    def energy_and_grad(self, q):
        """计算能量和梯度"""
        return prb_energy_and_grad(q, self.pr, self.B)
    
    def body_jacobian(self, q):
        """计算雅可比矩阵"""
        return prb_body_jacobian(q, self.pr)
    
    def body_jacobian_b(self, q):
        """计算关于磁场b的雅可比矩阵"""
        return prb_body_jacobian_b(q, self.pr)
    
    def space_jacobian(self, q):
        """计算空间雅可比矩阵"""
        return prb_space_jacobian(q, self.pr)
    
    def space_jacobian_b(self, q):
        """计算关于磁场b的空间雅可比矩阵"""
        return prb_space_jacobian_b(q, self.pr)

    def update_ex(self, q0=None, method='minimize', **kwargs):
        """优化更新配置 q"""
        if method == 'gradient':
            return prb_gradient_descent(self.pr, q0, **kwargs)
        elif method == 'minimize':
            return prb_update_ex(self.pr, self.B, q0)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
