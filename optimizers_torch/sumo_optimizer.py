import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class SUMO(Optimizer):
    """
    Implementation of SUMO: Subspace-Aware Moment-Orthogonalization.
    """

    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            beta: Tuple[float, float] = (0.9, 0.999),
            weight_decay: float = 1e-2,
            rank: int = 8,
            update_proj_gap: int = 200,
            alpha: float = 4.0,
            gamma: float = 1.1,
            moment_transformation=False,
            nesterov=True,
            momentum=0.95,
            norm_growth_limiter=True,
            gradient_perpendicular_scale: float = 1.0,
            lr_adam: float = 1e-5,
            weight_decay_adam: float = 0.0,
            eps: float = 1e-6,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta[0] < 1.0 or not 0.0 <= beta[1] < 1:
            raise ValueError(f"Invalid beta parameter: {beta}")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            alpha=alpha,
            gamma=gamma,
            moment_transformation=moment_transformation,
            momentum=momentum,
            nesterov=nesterov,
            norm_growth_limiter=norm_growth_limiter,
            gradient_perpendicular_scale=gradient_perpendicular_scale,
            lr_adam=lr_adam,
            weight_decay_adam=weight_decay_adam,
            eps=eps,
            correct_bias=correct_bias,
            no_deprecation_warning=no_deprecation_warning,
        )
        super().__init__(params, defaults)

        for g in params:
            if g["group_name"] == 'sumo_params':
                for p in g["params"]:
                    assert p.ndim == 2, p.ndim
                    self.state[p]["use_sumo"] = True
            else:
                for p in g["params"]:
                    self.state[p]["use_sumo"] = False

    @torch.compile
    @torch.no_grad()
    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            params = [p for p in group["params"] if self.state[p]["use_sumo"]]

            for p in params:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = None
                    state["norm_O_prev"] = None  # A scalar
                    state["proj_update"] = False

                state["step"] += 1

                # Adaptive Subspace Selection (Randomized SVD)
                # We update the subspace every K steps
                if "projector" not in state:
                    state["projector"] = SUMOProjector(group["rank"], update_proj_gap=group["update_proj_gap"],
                                                       scale=group["scale"], proj_type=group["proj_type"])

                # Projected Grad
                grad_hat = state["projector"].project(grad, state["step"])

                # Moment
                if state["m"] is None:
                    state["m"] = torch.zeros_like(grad_hat, device=grad_hat.device, dtype=grad_hat.dtype)
                state["m"].mul_(momentum).add_(grad_hat)

                # Nesterov
                if group["nesterov"]:
                    state["m"] = grad_hat.add(state["m"], alpha=momentum)

                # Low-Rank Steepest Descent (Moment Orthogonalization via SVD)
                # Exact SVD is used to avoid Newton-Schulz approximation errors [cite: 96, 203]
                U, _, Vh = torch.linalg.svd(state["m"], full_matrices=False)
                O = torch.matmul(U, Vh)
                del U
                del Vh

                # Norm-growth Limiter
                # Constrains abrupt increases in gradient norm
                if group["norm_growth_limiter"] and state["norm_O_prev"] is not None:
                    norm_curr = torch.norm(O)
                    norm_prev = state["norm_O_prev"]
                    if norm_curr / (norm_prev + 1e-8) > group["gamma"]:
                        O = (O / (norm_curr + 1e-8)) * group["gamma"] * norm_prev
                state["norm_O_prev"] = torch.norm(O)

                # Update weight in original space
                # Uses the projected orthogonal moment and the perpendicular gradient component
                # W = W - alpha*lr*(G - Q(scale*G_hat - adjust_lr*O)) - lr*wd*W
                adjust_lr_ratio = self.adjust_lr_for_muon(group["lr"], p.shape) / group["lr"]
                update_term = grad - state["projector"].project_back(
                    group["gradient_perpendicular_scale"] * grad_hat - adjust_lr_ratio * O)
                p.add_(update_term, alpha=-group["alpha"] * group["lr"])

                # Apply Weight Decay independently of moments
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                ############################
                #       AdamW backup       #
                ############################

            params = [p for p in group["params"] if not self.state[p]["use_sumo"]]

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["beta"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr_adam"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay_adam"] > 0.0:
                    p.add_(p, alpha=(-group["lr_adam"] * group["weight_decay_adam"]))

        return loss


class SUMOProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad) @ \
                            self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad) @ \
                             self.ortho_matrix[1].to(low_rank_grad.device.type)

        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type, Low_rank=False, niter=2):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        if Low_rank:  # randomized_svd
            q_ = min(2 * rank)
            U, s, Vh = torch.svd_lowrank(matrix, q=q_, niter=niter)
        else:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # make the smaller matrix always to be orthogonal matrix
        if type == 'right':
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')
