from typing import Callable

import torch
import torch.nn as nn
import warnings

class BGD:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
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
            lr: float = 1,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            dampening: float = 0.0,
            nesterov: bool = False,
            bounce_th: float = 0.7,
            second_moment: float = 1.,
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
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening: {dampening}")
        if bounce_th > 0.9:
            warnings.warn(f"Bounce TH set above 0.9: {bounce_th}. This may cause the optimizer to diverge due to infrequent lr reduction.", UserWarning)
        if second_moment < 1:
            raise ValueError(f"Invalid second_moment: {second_moment}. Must be >= 1 to ensure lr reduction.")

        self.model = model
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.device = self.trainable_params[0].device
        self.layer_lr = [lr for _ in range(len(self.trainable_params))]
        self.layer_2nd_moment = [second_moment for _ in range(len(self.trainable_params))]
        self.bounce_th = bounce_th
        self.non_blocking = non_blocking
        self._eps = _eps
        self._tolerance = _tolerance

    def bounce_update(self, idx: int, weight: torch.Tensor, oracle: torch.Tensor, gradient_weight: torch.Tensor, gradient_oracle: torch.Tensor) -> None:
        if (gradient_weight @ gradient_oracle) < 0:
            # print("bounce!")
            # Note: ".item()" on a GPU tensor would cause it to sync and transfer (copy) to host (CPU)
            f1, f2 = torch.linalg.vector_norm(gradient_weight), torch.linalg.vector_norm(gradient_oracle)
            s = f1 + f2
            if s < self._tolerance:
                d1, d2 = 0.5, 0.5
            else:
                s += torch.exp(-s) * self._eps
                d1, d2 = f2 / s, f1 / s

            if d1 > self.bounce_th:
                self.layer_2nd_moment[idx] += f2
                self.layer_lr[idx] /= self.layer_2nd_moment[idx]

            oracle.mul_(d2).add_(weight, alpha=d1)

        else:
            oracle.sub_(gradient_oracle.view_as(oracle), alpha=self.layer_lr[idx])

    def train(self,
              criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              train_loader: torch.utils.data.dataloader.DataLoader, ) -> None:
        self.model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            running_loss += loss.item()
            loss.backward()

            params_prev = [0] * len(self.trainable_params)
            grads_prev = params_prev.copy()
            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    params_prev[i] = p.clone()
                    grads_prev[i] = p.grad.view(-1).clone()
                    p.sub_(p.grad, alpha=self.layer_lr[i])
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, (p_prev, p_current, g_prev) in enumerate(zip(params_prev, self.trainable_params, grads_prev)):
                    self.bounce_update(i, p_prev, p_current, g_prev, p_current.grad.view(-1))
                    p_current.grad = None

        print(f"Train loss = {running_loss / len(train_loader):.3f}")

    @torch.no_grad()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> None:
        self.model.eval()
        correct = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum()

        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Accuracy = {accuracy.item():.2f}%")
