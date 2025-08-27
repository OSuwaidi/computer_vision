import warnings
from typing import Callable

import torch
import torch.nn as nn


class BGD:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): L2 penalty (default: 0), *coupled* (like classic SGD)
        bounce_th (float): threshold on d1 to trigger LR scaling (default: 0.9)
        _eps (float): epsilon for convex weights (default: 1e-12)
        _tolerance (float): maximum value before taking an equidistant weighted average
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 1,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            beta: float = 0.0,
            bounce_th: float = 0.7,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            _tolerance=1e-12,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self.params_prev: list[torch.Tensor] = []
        self.grads_prev: list[torch.Tensor] = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)
                self.params_prev.append(torch.empty_like(param))
                self.grads_prev.append(torch.empty_like(param))

        self.device = self.trainable_params[0].device
        self.lr = torch.tensor(lr, device=self.device)
        self.layer_v = torch.ones(len(self.trainable_params), device=self.device)
        self.beta = torch.tensor(beta, device=self.device)
        self.bounce_th = bounce_th
        self.non_blocking = non_blocking
        self._eps = torch.tensor(_eps, device=self.device)
        self._tolerance = _tolerance

    def bounce_update(self, idx: int, weight: torch.Tensor, oracle: torch.Tensor, gradient_weight: torch.Tensor, gradient_oracle: torch.Tensor) -> None:
        # Note: ".item()" on a GPU tensor would cause it to sync and transfer (copy) to host (CPU)
        # It's needed below to perform control flow logic on CPU rather than on GPU, then back on the CPU
        if (gradient_weight.view(-1) @ gradient_oracle.view(-1)).item() < 0:
            # print("bounce!")
            f1, f2 = torch.linalg.vector_norm(gradient_weight), torch.linalg.vector_norm(gradient_oracle)
            s = f1 + f2
            if not s.item() < self._tolerance:
                s += self._eps
                d1, d2 = f2 / s, f1 / s
            else:
                d1 = d2 = torch.tensor(0.5, device=self.device)

            if d1.item() > self.bounce_th:
                # TODO: should we use f2, s/2, **2, sqrt, etc.?
                self.layer_v[idx] += (self.beta * self.layer_v[idx] + (1. - self.beta) * f2)

            oracle.mul_(d2).add_(weight, alpha=d1)

        else:
            oracle.sub_(gradient_oracle * self.lr / self.layer_v[idx])

    def train(self,
              criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              train_loader: torch.utils.data.dataloader.DataLoader, ) -> None:
        self.model.train()
        running_loss = 0.
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            running_loss += loss.item()
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self.params_prev[i].copy_(p)
                    self.grads_prev[i].copy_(p.grad)
                    p.sub_(p.grad * self.lr / self.layer_v[i])
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, (p_prev, p_current, g_prev) in enumerate(zip(self.params_prev, self.trainable_params, self.grads_prev)):
                    self.bounce_update(i, p_prev, p_current, g_prev, p_current.grad)
                    p_current.grad = None

        print(f"Train loss = {running_loss / len(train_loader):.3f}")

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> None:
        self.model.eval()
        correct = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum()

        accuracy = (100. * correct / len(test_loader.dataset)).item()
        print(f"Accuracy = {accuracy:.2f}%")
        return accuracy
