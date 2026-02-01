# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from accelerate.logging import get_logger
from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor

logger = get_logger(__name__)
class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr=3e-5,
            wd=0.1,  # muon
            weight_decay: float = 0.0,  # galore
            # muon_params=None,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_params=None,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=betas,
            adamw_eps=eps,
            correct_bias=correct_bias,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)
        for g in params:
            if g["group_name"] == 'muon_params':
                for p in g["params"]:
                    assert p.ndim == 2, p.ndim
                    self.state[p]["use_muon"] = True
            elif g["group_name"] == 'galore_params':
                for p in g["params"]:
                    self.state[p]["use_muon"] = False
            elif g["group_name"] == 'regular_params':
                for p in g["params"]:
                    self.state[p]["use_muon"] = False
            else:
                raise RuntimeError("Param not associated")

    @torch.compile
    @torch.no_grad()
    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if 'dim' not in group:
                    group['dim'] = 2

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group['dim'] <= 2:
                            state["projector"] = GaLoreProjector(group["rank"],
                                                                 update_proj_gap=group["update_proj_gap"],
                                                                 scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"],
                                                                       update_proj_gap=group["update_proj_gap"],
                                                                       scale=group["scale"],
                                                                       proj_type=group["proj_type"])
                    g_ = g.clone()
                    g = state["projector"].project(g, state["step"])

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                # u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                u, _, v = torch.linalg.svd(g, full_matrices=False)
                # condition_number = float('inf') if s.min().item() == 0 else (s.max() / s.min()).item()
                # logger.info(f"Condition number: {condition_number}")
                u = u @ v  # torch.matmul(u, v.mT)
                # if g.shape[1]>g.shape[0]:
                #     u = chebyshev_nearest_orthogonal_projection_torch(g.mT).mT
                # else:
                # u = orthogonal_projection_halley(g)
                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)
                g_per = g_ - state["projector"].project_back(state["projector"].project(g_, state["step"]))
                # GaLore Projection Back
                if "rank" in group:
                    u = state["projector"].project_back(u)
                # apply update
                p.data.add_(u + g_per, alpha=-adjusted_lr)
                state["step"] += 1

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if g.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if 'dim' not in group:
                    group['dim'] = 2

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group['dim'] <= 2:
                            state["projector"] = GaLoreProjector(group["rank"],
                                                                 update_proj_gap=group["update_proj_gap"],
                                                                 scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"],
                                                                       update_proj_gap=group["update_proj_gap"],
                                                                       scale=group["scale"],
                                                                       proj_type=group["proj_type"])
                    g = state["projector"].project(g, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(g)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(g)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["adamw_betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(g, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["adamw_eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                if group["wd"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["wd"]))

        return loss

# class AdamW(Optimizer):
#     """
#     Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
#     Regularization](https://arxiv.org/abs/1711.05101).
#
#     Parameters:
#         params (`Iterable[nn.parameter.Parameter]`):
#             Iterable of parameters to optimize or dictionaries defining parameter groups.
#         lr (`float`, *optional*, defaults to 0.001):
#             The learning rate to use.
#         betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
#             Adam's betas parameters (b1, b2).
#         eps (`float`, *optional*, defaults to 1e-06):
#             Adam's epsilon for numerical stability.
#         weight_decay (`float`, *optional*, defaults to 0.0):
#             Decoupled weight decay to apply.
#         correct_bias (`bool`, *optional*, defaults to `True`):
#             Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
#         no_deprecation_warning (`bool`, *optional*, defaults to `False`):
#             A flag used to disable the deprecation warning (set to `True` to disable the warning).
#     """
#
#     def __init__(
#         self,
#         params: Iterable[nn.parameter.Parameter],
#         lr: float = 1e-3,
#         betas: Tuple[float, float] = (0.9, 0.999),
#         eps: float = 1e-6,
#         weight_decay: float = 0.0,
#         correct_bias: bool = True,
#         no_deprecation_warning: bool = False,
#     ):
#         if not no_deprecation_warning:
#             warnings.warn(
#                 "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
#                 " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
#                 " warning",
#                 FutureWarning,
#             )
#         require_version("torch>=1.5.0")  # add_ with alpha
#         if lr < 0.0:
#             raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
#         if not 0.0 <= eps:
#             raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
#         defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
#         super().__init__(params, defaults)
#
#     @torch.no_grad()
#     def step(self, closure: Callable = None):
#         """
#         Performs a single optimization step.
#
#         Arguments:
#             closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
#
#                 state = self.state[p]
#
#                 if "step" not in state:
#                     state["step"] = 0
#
#                 if 'dim' not in group:
#                     group['dim'] = 2
#
#                 # GaLore Projection
#                 if "rank" in group:
#                     if "projector" not in state:
#                         if group['dim'] <=2:
#                             state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
#                         else:
#                             state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
#                     grad = state["projector"].project(grad, state["step"])
#
#                 # State initialization
#                 if "exp_avg" not in state:
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(grad)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(grad)
#
#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]
#
#                 state["step"] += 1
#
#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])
#
#                 step_size = group["lr"]
#                 if group["correct_bias"]:  # No bias correction for Bert
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
#
#                 # compute norm gradient
#                 norm_grad = exp_avg / denom
#
#                 # GaLore Projection Back
#                 if "rank" in group:
#                     norm_grad = state["projector"].project_back(norm_grad)
#
#                 p.add_(norm_grad, alpha=-step_size)
#
#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 # Add weight decay at the end (fixed version)
#                 if group["weight_decay"] > 0.0:
#                     p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
#
#         return loss
