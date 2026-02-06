from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from math import sqrt

from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SelfBGDv:  # BETTER THAN SelfBGDm
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
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,

    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_P: torch.Tensor = torch.empty_like(params_vec)
        self._v: torch.Tensor = torch.zeros_like(params_vec)

        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.b = beta
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps

    @torch.no_grad()
    def _bounce(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()

        self._v.mul_(self.b).add_(G)

        if (self._v @ G) < 0.:
            w = G.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            P.lerp_(self._prev_P, weight=w)

        else:
            self._prev_P.copy_(P)

            P.sub_(self._v, alpha=self.lr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

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
            self._bounce()  # compares inter-batch gradients

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


class SelfBGDm:
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
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_P: torch.Tensor = torch.empty_like(params_vec)
        self._m: torch.Tensor = torch.zeros_like(params_vec)

        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.b = beta
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.t = 0

    @torch.no_grad()
    def _bounce(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()
        self._m.lerp_(G, weight=(1. - self.b))

        if (self._m @ G) < 0.:
            w = G.abs_().sub_(self._m.abs_()).sigmoid_()
            self._m.zero_()
            self.t = 0
            P.lerp_(self._prev_P, weight=w)

        else:
            self._prev_P.copy_(P)

            self.t += 1
            b_corr = 1. - self.b ** self.t
            P.sub_(self._m, alpha=self.lr / b_corr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

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
            self._bounce()  # compares inter-batch gradients

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


class BGDNew:
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
        non_blocking: bool = False,
        _eps: float = 1e-8,
        beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_P: torch.Tensor = torch.empty_like(params_vec)
        self._v: torch.Tensor = torch.zeros_like(params_vec)
        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.b = beta
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps

    @torch.no_grad()
    def _bounce1(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()

        self._v.mul_(self.b).add_(G)

        if (self._v @ G) < 0.:
            w = G.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            P.lerp_(self._prev_P, weight=w)
        else:
            self._prev_P.copy_(P)
            P.sub_(self._v, alpha=self.lr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _bounce2(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()

        if (self._v @ G) < 0.:
            w = G.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            P.lerp_(self._prev_P, weight=w)
        else:
            self._prev_P.copy_(P)
            self._v.mul_(self.b).add_(G)
            P.sub_(self._v, alpha=self.lr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        self.model.train()
        epoch_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(self.device, non_blocking=self.non_blocking)
            y = y.to(self.device, non_blocking=self.non_blocking)

            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz

            loss.backward()
            self._bounce1()

            criterion(self.model(x), y).backward()
            self._bounce2()

        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0

        for x, y in test_loader:
            x = x.to(self.device, non_blocking=self.non_blocking)
            y = y.to(self.device, non_blocking=self.non_blocking)

            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        return 100.0 * correct / n_samples


from typing import Callable
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BGDCent:  # TODO: Try Global Batch BGD
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
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_params: torch.Tensor = torch.empty_like(params_vec)
        self._v: torch.Tensor = torch.zeros_like(params_vec)

        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta

    @torch.no_grad()
    def _bounce_update(self, curr_p: torch.Tensor) -> None:
        curr_grad = self._get_grads_vec()  # re-calculate gradients at the new position
        if (self._v @ curr_grad) < 0.:
            # print("bounce!")
            curr_grad.abs_().sub_(curr_grad.mean())
            curr_grad.abs_().sub_(curr_grad.mean())
            w = curr_grad.abs_().sub_(self._v.abs_()).sigmoid_()

            self._v.zero_()

            curr_p.lerp_(self._prev_params, weight=w)

        else:
            curr_p.sub_(curr_grad, alpha=self.lr)  # TODO: div this lr by 2?

        vector_to_parameters(curr_p, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

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

            # --- Preliminary Update ---
            with torch.no_grad():
                P = parameters_to_vector(self.trainable_params)
                G = self._get_grads_vec()
                self._prev_params.copy_(P)

                # Update velocity (momentum)
                v = self._v.mul_(self.b).add_(G)

                # Apply update
                P.sub_(v, alpha=self.lr)

                # PUSH BACK: Write the updated flat params into the actual model layers
                vector_to_parameters(P, self.trainable_params)

                self.model.zero_grad(set_to_none=True)

            # --- Second Forward/Backward Pass (Lookahead) ---
            criterion(self.model(x), y).backward()

            self._bounce_update(P)

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
