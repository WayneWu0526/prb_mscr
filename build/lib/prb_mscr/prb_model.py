from .forward import prb_forward
from .update_ex import prb_update_ex
from .update_gradient import gradient_descent
from .params import set_param_function

class PRBModel:
    def __init__(self, pr, B):
        self.pr = pr
        self.B = B

    @classmethod
    def initialize(cls, L, E, r, nu, B_norm, L_rigid, Nm, N, m):
        """
        初始化 PRBModel 的参数配置。
        """
        pr = set_param_function(L, E, r, nu, B_norm, L_rigid, Nm, N, m)
        B = B_norm * [0, 0, 1]
        return cls(pr, B)

    def forward(self, q):
        """计算正向运动学"""
        return prb_forward(q, self.pr)

    def optimize(self, q0=None, method='gradient', **kwargs):
        """优化更新配置 q"""
        if method == 'gradient':
            return gradient_descent(self.pr, self.B, q0, **kwargs)
        elif method == 'minimize':
            return prb_update_ex(self.pr, self.B, q0)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
