from typing import Callable
import torch
import torch.nn as nn


class Adam:
    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.001,
            non_blocking: bool = False,
            _eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self.first_moment: list[torch.Tensor] = []
        self.second_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self.first_moment.append(torch.zeros_like(param))
                self.second_moment.append(torch.zeros_like(param))

        self.lr = lr
        self.device = param.device
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b1 = 0.9
        self.b2 = 0.999
        self.t = 0

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

            with torch.no_grad():
                b1_corr = 1. - self.b1 ** self.t
                b2_corr = 1. - self.b2 ** self.t

                for i, p in enumerate(self.trainable_params):
                    # Note that m and v here are references (aliases) to the actual parameter in memory (same underlying storage)
                    m = self.first_moment[i].lerp_(p.grad, weight=(1. - self.b1))
                    v = self.second_moment[i].lerp_(p.grad.square(), weight=(1. - self.b2))
                    denom = v.div(b2_corr).sqrt_().add_(self._eps)  # one temporary tensor creation (allocation)
                    p.addcdiv_(m, denom, value=-self.lr/b1_corr)  # we moved "m.div(b1_corr)" to lr to avoid temp tensor allocation
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


class SGDMomentum:
    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.01,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params = []
        self.v = []
        for p in model.parameters():
            if p.requires_grad:
                self.trainable_params.append(p)
                self.v.append(torch.zeros_like(p))

        self.lr = lr
        self.device = self.trainable_params[0].device
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.beta = beta

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

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    v = self.v[i].mul_(self.beta).add_(p.grad)
                    p.sub_(v, alpha=self.lr)
                    p.grad = None

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


class check_SGD:
    """
    Store all intermediate activations during forward pass, then backward.
    Per layer, from right (deep) to left (shallow), take a step (update) then forward based on previous activations to analyze the loss.
    For layer k, if after stepping (param update) and forward from it, the loss decreased, then keep updates and continue, else either:
        1. take smaller step
        2. step in opposite direction
        3. skip
    Then continue, and repeat for other layers.
    """
    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.01,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params = []
        self.v = []
        for p in model.parameters():
            if p.requires_grad:
                self.trainable_params.append(p)
                self.v.append(torch.zeros_like(p))

        self.lr = lr
        self.device = self.trainable_params[0].device
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.t = 0
        self.beta = beta

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

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    v = self.v[i].mul_(self.beta).add_(p.grad)
                    p.sub_(v, alpha=self.lr)
                    p.grad = None

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
