import math
from typing import Callable, Iterable, Optional
import torch
from torch.optim.optimizer import Optimizer


import math
from typing import Callable, Iterable, Optional
import torch
from torch.optim.optimizer import Optimizer

class BounceSGD(Optimizer):
    r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): base learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): L2 penalty (default: 0), *coupled* (like classic SGD)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        bounce_th (float): threshold on d1 to trigger LR scaling (default: 0.9)
        lr_scale (float): factor to divide LR by when bounce triggers (default: 1.1)
        _eps (float): epsilon for convex weights (default: 1e-12)
        _tolerance (float): maximum value before taking an equidistant weighted average
    """
    def __init__(
        self,
        params: Iterable,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        bounce_th: float = 0.9,
        lr_scale: float = 1.1,
        _eps: float = 1e-8,
        _tolerance = 1e-12
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening: {dampening}")
        if lr_scale <= 0.0:
            raise ValueError(f"Invalid lr_scale: {lr_scale}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        # ---- Phase 0: snapshot params and grads (before base step)
        snapshots = []  # per-group list of (param, prev_param_copy, prev_grad_flat)
        for group in self.param_groups:
            group_snaps = []
            for p in group["params"]:
                if p.grad is None:
                    group_snaps.append((p, None, None))
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("BounceSGD does not support sparse gradients")
                # true copies (no aliasing)
                prev_p = p.clone()
                prev_g = p.grad.clone().reshape(-1)
                group_snaps.append((p, prev_p, prev_g))
                # init momentum state
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if group["momentum"] != 0 and "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
            snapshots.append(group_snaps)

        # ---- Phase 1: base SGD(-like) step
        for group, group_snaps in zip(self.param_groups, snapshots):
            lr = group["lr"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            damp = group["dampening"]
            nesterov = group["nesterov"]

            for (p, _, _) in group_snaps:
                grad = p.grad
                if grad is None:
                    continue
                # classic L2 (coupled) decay
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                state = self.state[p]
                state["step"] += 1

                if mom != 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(grad, alpha=1 - damp)
                    d_p = grad.add(buf, alpha=mom) if nesterov else buf
                else:
                    d_p = grad

                p.add_(d_p, alpha=-lr)

        # ---- Phase 2: recompute grads via closure (required for bounce)
        loss = None
        if closure is None:
            raise RuntimeError("BounceSGD requires a closure for the second-phase gradients.")
        loss = closure()  # user should zero grads inside the closure

        # ---- Phase 3: bounce / second-step logic
        # Accumulate whether to scale LR for each group; apply once per group.
        scale_group = [False] * len(self.param_groups)

        for gi, (group, group_snaps) in enumerate(zip(self.param_groups, snapshots)):
            th = group["bounce_th"]
            eps = group["convex_eps"]
            second_lr = group["second_lr"] if group["second_lr"] is not None else group["lr"]

            for (p, prev_p, prev_g) in group_snaps:
                if p.grad is None or prev_p is None or prev_g is None:
                    continue

                g_now = p.grad.reshape(-1)
                # dot test
                if torch.dot(prev_g, g_now) < 0:
                    # convex weights
                    n1 = prev_g.norm()
                    n2 = g_now.norm()
                    s = n1 + n2
                    if s <= eps:
                        d1 = d2 = torch.tensor(0.5, dtype=p.dtype, device=p.device)
                    else:
                        d1 = n2 / s  # weight on prev param
                        d2 = n1 / s  # weight on current param

                    # convex combine: p = d2 * p + d1 * prev_p
                    p.mul_(d2).add_(prev_p, alpha=d1)

                    if (d1 > th) or (d2 > th):
                        scale_group[gi] = True
                else:
                    # optional extra plain step on the new gradient
                    p.add_(p.grad, alpha=-second_lr)

        # ---- Phase 4: apply LR scaling once per group (if requested)
        for gi, group in enumerate(self.param_groups):
            if scale_group[gi]:
                group["lr"] = group["lr"] / group["lr_scale"]

        return loss
