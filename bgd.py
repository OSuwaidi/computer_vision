from typing import Callable, Iterable, Optional
import torch
import torch.nn as nn
from tqdm.auto import trange
# from torch.optim.optimizer import Optimizer


class BounceSGD:
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
        model: nn.Module,
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
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening: {dampening}")

        self.model = model
        self.params = model.parameters()
        self.num_params = len(tuple(self.params))
        self.device = model.device
        self.lr = lr
        self.bounce_th = bounce_th
        self.lr_scale = lr_scale
        self._eps = _eps
        self._tolerance = _tolerance

    def dist(self, g1: torch.Tensor, g2: torch.Tensor) -> tuple:
        # Note: `.item()` on a GPU tensor would cause it to sync and transfer (copy) to host (CPU)
        f1, f2 = torch.linalg.vector_norm(g1), torch.linalg.vector_norm(g2)
        s = f1 + f2
        if s < self._tolerance:
            return 0.5, 0.5

        s += torch.exp(-s) * self.eps
        return f2 / s, f1 / s

    def bounce_update(self, weight: torch.Tensor, oracle: torch.Tensor, gradient_weight: torch.Tensor, gradient_oracle: torch.Tensor, optimizer) -> int:
        extremes = 0
        if (gradient_weight @ gradient_oracle) < 0:
            # print("bounce!")
            d1, d2 = self.dist(gradient_weight, gradient_oracle)
            if d1 > self.bounce_th:
                extremes += 1

            oracle.mul_(d2).add_(weight, alpha=d1)

        else:
            oracle.sub_(gradient_oracle.view_as(oracle), alpha=self.lr)

        return extremes

    # @torch.no_grad()
    # def step(self):
    #     params_current = [0] * self.num_params
    #     grads_current = params_current.copy()
    #
    #     for i, p in enumerate(self.params):
    #         params_current[i], grads_current[i] = p.clone(), p.grad.clone()
    #         p.sub_(p.grad, alpha=self.lr)

    def train(self,
              criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
              epochs: int):

        for _ in trange(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                loss = criterion(self.model(x), y)
                loss.backward()

                with torch.no_grad():
                    params_current = [0] * self.num_params
                    grads_current = params_current.copy()

                    for i, p in enumerate(self.params):
                        if p.requires_grad:
                            params_current[i], grads_current[i] = p.clone(), p.grad.clone()
                            p.sub_(p.grad, alpha=self.lr)
                            p.grad = None

                criterion(self.model(x), y).backward()




