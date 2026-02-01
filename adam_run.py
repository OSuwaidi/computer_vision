# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import torch
from adam_opt import Adam
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50, wide_resnet50_2
import torch.nn as nn
from tqdm import trange
import numpy as np
import torch.nn.functional as F
import random
import wandb


m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
fill = [int(255 * x) for x in m]
norm = v2.Normalize(mean=m, std=s)
T_train = v2.Compose([
    v2.PILToTensor(),
    v2.RandomCrop(32, padding=4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),
    # v2.RandAugment(num_ops=2, magnitude=9, fill=fill),
    v2.TrivialAugmentWide(),
    v2.ToDtype(torch.float32, scale=True),
    norm,
    v2.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
])
T_test = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    norm,
])
train_data = datasets.CIFAR10(root=f'../datasets/CIFAR10', download=True, train=True, transform=T_train)
test_data = datasets.CIFAR10(root=f'../datasets/CIFAR10', train=False, transform=T_test)

DEVICE = "cuda"
BS: int = 128
PIN_MEM: bool = True
N_workers = 4
PERSIST = True if N_workers > 0 else False


def run_adam(model, epochs=150, seed=0, lr=0.001, use_bias=False):

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    best_acc = 0.
    min_loss = float("inf")

    group_name = f"adam-{model}-{lr}"
    run = wandb.init(
        project="bgd-cifar10",
        group=group_name,
        name=f"seed_{seed}",
        tags=[str(lr), model, "bias" if use_bias else "nobias"],
        config={
            "bgd_type": "adam",
            "lr": lr,
            "epochs": epochs,
            "model": model,
            "batch_size": BS,
            "seed": seed,
            "bias": use_bias,
        },
    )
    run.define_metric("acc", summary="max")
    run.define_metric("loss", summary="min")

    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if model == "resnet18":
        m = resnet18()
        m.fc = nn.Linear(512, 10, bias=use_bias)
    elif model == "resnet50":
        m = resnet50()
        m.fc = nn.Linear(2048, 10, bias=use_bias)
    elif model == "wide_resnet50":
        m = wide_resnet50_2()
        m.fc = nn.Linear(2048, 10, bias=use_bias)
    else:
        raise ValueError(f"Unknown model: {model}")

    m = m.to(DEVICE)

    opt = Adam(m, lr=lr, non_blocking=PIN_MEM)

    train_loader = DataLoader(train_data,
                              batch_size=BS,
                              num_workers=N_workers,
                              persistent_workers=PERSIST,
                              drop_last=True,
                              shuffle=True,
                              pin_memory=PIN_MEM,
                              generator=torch.Generator().manual_seed(seed),
                              worker_init_fn=seed_worker
                              )
    test_loader = DataLoader(test_data, batch_size=BS, num_workers=N_workers, pin_memory=PIN_MEM)

    for epoch in trange(epochs):
        loss = opt.train(train_loader, F.cross_entropy)
        acc = opt.test(test_loader)

        best_acc = max(best_acc, acc)
        min_loss = min(min_loss, loss)
        run.log({"loss": loss, "acc": acc}, commit=True, step=epoch)

    run.summary["best_acc"] = best_acc
    run.summary["min_loss"] = min_loss

    artifact = wandb.Artifact(name=group_name + f"-{seed}", type="model")
    with artifact.new_file(f"{group_name}_ckpt_{seed}.pt", mode="wb") as f:
        torch.save(m.state_dict(), f)

    run.log_artifact(artifact).wait()
    run.finish()


if __name__ == "__main__":
    seeds = (192, 169, 420,)
    lrs = (0.1, 0.01, 0.001)
    models = ("resnet18", "resnet50", "wide_resnet50")
    for model in models:
        for lr in lrs:
            print(f"Running {model} on Adam using lr: {lr}\n")
            for seed in seeds:
                run_adam(model=model, epochs=150, seed=seed, lr=lr)
