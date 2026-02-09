"""
Muon optimizer implementation.

Muon is a specialized optimizer for training large language models that
provides better training stability and faster convergence compared to AdamW.

Key features:
- Separate learning rates for different parameter groups
- Momentum-based updates for 2D parameters (matrices)
- Adam-style updates for embeddings and 1D parameters
- Designed to work well with large learning rates

Based on the Muon optimizer from the Modded NanoGPT project.
"""

import torch
import torch.optim as optim
from typing import Optional, Iterable


class Muon(optim.Optimizer):
    """
    Muon optimizer for training large language models.

    Muon uses different update strategies for different parameter types:
    - 2D parameters (matrices): Momentum-based SGD with Nesterov acceleration
    - 1D parameters (biases, norms): Adam-style updates
    - Embeddings: Adam-style updates

    This separation allows for more aggressive learning rates on the
    large weight matrices while maintaining stability on other parameters.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 3e-4)
        momentum: Momentum factor (default: 0.95)
        nesterov: Enable Nesterov momentum (default: True)
        adam_beta1: Adam beta1 for 1D/embedding params (default: 0.9)
        adam_beta2: Adam beta2 for 1D/embedding params (default: 0.999)
        adam_eps: Adam epsilon for numerical stability (default: 1e-8)
        wd: Weight decay (default: 0.1)
        lr_1d: Learning rate multiplier for 1D/embedding params (default: 0.1)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        wd: float = 0.1,
        lr_1d: float = 0.1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if wd < 0.0:
            raise ValueError(f"Invalid weight decay value: {wd}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
            wd=wd,
            lr_1d=lr_1d,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["wd"]
            lr = group["lr"]
            lr_1d = group["lr_1d"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Determine parameter type and update strategy
                is_2d = p.ndim == 2
                is_embedding = group.get("is_embedding", False)

                if is_2d and not is_embedding:
                    # --- 2D parameters: Momentum SGD with weight decay ---
                    # This is the core "Muon" update for large weight matrices

                    # Initialize momentum buffer
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]

                    # Apply weight decay (L2 regularization)
                    if wd != 0:
                        grad = grad.add(p, alpha=wd)

                    # Update momentum buffer
                    buf.mul_(momentum).add_(grad)

                    # Apply Nesterov momentum if enabled
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf

                    # Apply gradient step
                    p.add_(update, alpha=-lr)

                else:
                    # --- 1D/embedding parameters: Adam-style update ---
                    # More conservative learning rate for these parameters

                    # Initialize Adam state
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["adam_beta1"], group["adam_beta2"]
                    eps = group["adam_eps"]

                    state["step"] += 1
                    step = state["step"]

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # Apply weight decay (optional for embeddings)
                    if wd != 0:
                        grad = grad.add(p, alpha=wd)

                    # Update exponential moving averages
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute denominator with bias correction
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                    # Compute step size with bias correction
                    step_size = lr_1d * lr / bias_correction1

                    # Apply Adam update
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "muon",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    **kwargs
) -> optim.Optimizer:
    """
    Get an optimizer instance for training.

    Args:
        model: The model to optimize
        optimizer_name: Either 'muon' or 'adamw'
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer instance
    """
    if optimizer_name.lower() == "muon":
        embedding_param_ids = set()
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                for p in module.parameters(recurse=False):
                    embedding_param_ids.add(id(p))

        embedding_params = []
        one_d_params = []
        two_d_params = []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in embedding_param_ids:
                embedding_params.append(p)
            elif p.ndim < 2:
                one_d_params.append(p)
            else:
                two_d_params.append(p)

        param_groups = []
        if two_d_params:
            param_groups.append({"params": two_d_params})
        if one_d_params:
            param_groups.append({"params": one_d_params, "wd": 0.0})
        if embedding_params:
            param_groups.append({"params": embedding_params, "is_embedding": True, "wd": 0.0})

        return Muon(
            param_groups if param_groups else model.parameters(),
            lr=learning_rate,
            wd=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'muon' or 'adamw'")
