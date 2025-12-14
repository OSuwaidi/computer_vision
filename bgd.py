from typing import Callable

import torch
import torch.nn as nn


class BGD:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        weight_decay (float, optional): L2 penalty (default: 0), *coupled* (like classic SGD)
        bounce_th (float): threshold on d1 to trigger LR scaling (default: 0.7)
        _eps (float): epsilon for convex weights (default: 1e-12)
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            lr: float = 1,
            weight_decay: float = 0.0,
            use_second_moment: bool = False,
            beta: float = 0.999,
            bounce_th: float = 0.7,
            non_blocking: bool = False,
            _eps: float = 1e-9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"Invalid beta: {beta}")

        self.model = model
        self.criterion = criterion
        self.trainable_params: list[nn.Parameter] = []
        self._params_prev: list[torch.Tensor] = []
        self._grads_prev: list[torch.Tensor] = []
        self.second_moment: list[torch.Tensor] = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)
                self._params_prev.append(torch.empty_like(param))
                self._grads_prev.append(torch.empty_like(param))
                if use_second_moment:
                    self.second_moment.append(torch.zeros_like(param))

        self.device = self.trainable_params[0].device
        self.layer_scale: torch.Tensor = torch.ones(len(self.trainable_params), device=self.device)  # per-layer (not per-param) adaptive learning rate
        self.lr = lr
        self._t: int = 0
        self.beta = beta
        self.bounce_th = bounce_th
        self.non_blocking = non_blocking
        self._eps = _eps
        self.get_second_moment = self._get_second_moment if use_second_moment else lambda *_: 1.

    def _get_second_moment(self, idx, gradient: torch.Tensor) -> torch.Tensor:
        # TODO: experiment with corrected unbiased second moment (should only be computed after the first iteration) (v_t - B^t * c) / (1 - B^t)
        self.second_moment[idx].mul_(self.beta).add_(gradient.abs(), alpha=1 - self.beta)  # update second moment
        return (self.second_moment[idx] - self.beta ** self._t) / (1. - self.beta ** self._t)  # unbiased second moment

    def _bounce_update(self, idx: int, weight: torch.Tensor, oracle: torch.Tensor, gradient_weight: torch.Tensor, gradient_oracle: torch.Tensor) -> None:
        # Note: ".item()" on a GPU tensor would cause it to sync and transfer (copy) to host (CPU)
        # It's needed below to perform control flow logic on CPU rather than on GPU, then back on the CPU
        if (gradient_weight.view(-1) @ gradient_oracle.view(-1)).item() < 0.:
            # print("bounce!")
            f1, f2 = torch.linalg.vector_norm(gradient_weight), torch.linalg.vector_norm(gradient_oracle)
            s = f1 + f2
            if not s.item() < self._eps:
                s.add_(self._eps)
                d1, d2 = f2 / s, f1 / s
            else:
                d1 = d2 = 0.5

            if d1.item() > self.bounce_th:
                # TODO: should we use f2, s/2, **2, sqrt, etc.?
                self.layer_scale[idx].add_(f2)

            oracle.mul_(d2).add_(weight, alpha=d1)

        else:
            second_moment_unbias = self.get_second_moment(idx, gradient_oracle)
            denom = self.layer_scale[idx] * second_moment_unbias + self._eps
            oracle.addcdiv_(gradient_oracle, denom, value=-self.lr)

    @staticmethod
    def _bn_eval(bn: nn.Module) -> None:
        if isinstance(bn, nn.modules.batchnorm._BatchNorm):
            bn.train(False)  # same as bn.eval()

    @staticmethod
    def _bn_train(bn: nn.Module) -> None:
        if isinstance(bn, nn.modules.batchnorm._BatchNorm):
            bn.train(True)

    def train(self,
              train_loader: torch.utils.data.DataLoader, ) -> float:
        self.model.train()
        epoch_loss = 0.
        # TODO: set a lower bound on the minimum possible lr (e.g. 0.0001)?
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = self.criterion(self.model(x), y)
            epoch_loss += loss.item()
            loss.backward()
            self._t += 1

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._params_prev[i].copy_(p)
                    self._grads_prev[i].copy_(p.grad)
                    second_moment_unbias = self.get_second_moment(i, p.grad)
                    denom = self.layer_scale[i] * second_moment_unbias + self._eps
                    p.addcdiv_(p.grad, denom, value=-self.lr)
                    p.grad = None

            self.model.apply(self._bn_eval)
            self.criterion(self.model(x), y).backward()
            self.model.apply(self._bn_train)
            self._t += 1

            with torch.no_grad():
                for i, (p_prev, p_current, g_prev) in enumerate(zip(self._params_prev, self.trainable_params, self._grads_prev)):
                    self._bounce_update(i, p_prev, p_current, g_prev, p_current.grad)
                    p_current.grad = None

        return epoch_loss / len(train_loader)

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum()

        accuracy = (100. * correct / len(test_loader.dataset)).item()
        return accuracy
