
from typing import Callable
import torch
import torch.nn as nn
import numpy as np


class BGD:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            beta: float = 0.9,
            _eps: float = 1e-8,
            non_blocking: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # store reference (pointer) to actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = np.zeros(len(self.trainable_params), dtype=np.uint32)

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._first_moment[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)) < 0.:
            # print("bounce!")
            w = curr_grad.abs().sub_(prev_grad.abs()).sigmoid_()

            mixed_grads = curr_grad.lerp_(prev_grad, weight=w)
            prev_grad.zero_().lerp_(mixed_grads, weight=(1. - self.b))
            self.t[idx] = 1

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            # --- Preliminary Update ---
            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)

                    # Update moment
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))

                    # Bias correction
                    b_corr = 1. - self.b ** self.t[i].item()

                    # Apply update
                    p.sub_(m, alpha=self.lr/b_corr)  # TODO: WHAT IF WE MOVE BASED ON g, NOT m (USE m TO POPULATE ONLY)???!!! AND WHAT IF WE USE POLYAK MOMENTUM HERE?
                    p.grad = None

            # --- Second Forward/Backward Pass (Lookahead) ---
            # Re-calculate gradients at the new position
            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy