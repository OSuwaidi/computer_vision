import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import shutil
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce, ReduceOp, get_world_size
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from models.FR import FaceNet

# This entire script will launch/run for every GPU (process) individually!

def fix_facecrub(facescrub_path: Path):
    # Move all person's image folders into one unified root directory:
    for genders in facescrub_path.iterdir():
        for person in genders.iterdir():
            shutil.move(person, facescrub_path)
        genders.rmdir()

world_size = torch.cuda.device_count()  # total number of processes
# Each process (GPU) will receive this complete BS ==> effective BS = num_GPUs * BS
# Hence, to get the desired gradient BS, make sure you divide by the number of GPUs
BS = 2 ** 7 // world_size
NUM_WORKERS = world_size * 4  # 4 workers per GPU
EPOCHS = 5
LR = 1e-3
WD = 5e-5
SEED = 42
device = "cuda"
torch.backends.cudnn.benchmark = True
CEL = nn.CrossEntropyLoss()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
generator = torch.Generator().manual_seed(SEED)


def train(gpu_id: int, epochs, model, train_loader, optimizer, scheduler):
    model.train()
    iters_per_epoch = len(train_loader)  # number (not size) of batches processed per epoch
    total_samples = train_loader.batch_size * get_world_size() * iters_per_epoch  # "get_world_size()" only works AFTER initializing default process group

    for e in range(1, epochs + 1):
        loop = tqdm(train_loader, desc=f"Epoch {e}", disable=(gpu_id != 0))  # only print on GPU 0
        train_loader.sampler.set_epoch(e)  # this ensures data shuffling (randomization) for different epochs

        epoch_loss = torch.zeros((), device=gpu_id)  # running epoch loss
        correct = torch.zeros((), device=gpu_id, dtype=torch.int32)
        for x, y in loop:
            x, y = x.to(gpu_id, non_blocking=True), y.to(gpu_id, non_blocking=True)
            embedding = model(x)
            logits = model.module.classify(embedding, y)
            loss = CEL(logits, y)  # rank-local loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # DDP all-reduces gradients via AVG operation across processes (resulting in gradient for global BS)
            optimizer.step()
            scheduler.step()

            with torch.inference_mode():
                W = F.normalize(model.module.classifier.weight, p=2, dim=1)
                true_logits = F.linear(embedding, W)  # no margin logits
                pred = true_logits.argmax(1)  # highest classification class

                epoch_loss += loss
                correct += pred.eq(y).sum()

        with torch.inference_mode():
            # Collective calls must run on all ranks:
            reduce(epoch_loss, dst=0, op=ReduceOp.AVG)
            reduce(correct, dst=0, op=ReduceOp.SUM)

        if gpu_id == 0:
            print(f"Epoch {e}/{EPOCHS} | loss: {epoch_loss / iters_per_epoch:.4} | Acc: {correct / total_samples:.2%}")

    final_loss = epoch_loss.item() / iters_per_epoch
    final_acc = correct.item() / total_samples * 100.
    return final_loss, final_acc


@torch.inference_mode()
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)  # defaults to device cuda:0
        embedding = model(x)
        pred = model.module.classify(embedding, y).argmax(1)
        correct += pred.eq(y).sum()
        total += y.numel()

    acc = correct.item() / total
    print(f"Top-1 Test Accuracy: {acc:.2%}")
    return acc * 100.


def setup_ddp(rank: int) -> None:
    # rank ∈ [0, world_size-1] --> unique process/GPU id
    torch.cuda.set_device(rank)  # to ensure that each process runs exclusively on a single GPU
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # To get default supported backend: "torch.distributed.get_default_backend_for_device(device_type)"


# Runs for each child process (rank). Any function used below must be defined on global scope (accessible to all processes). Below you define per-process procedures.
def main(rank: int, train_dataset, test_dataset):
    setup_ddp(rank)
    print(f"setup complete for gpu# {rank}.")
    train_loader = DataLoader(
        train_dataset,
        BS,  # per GPU BS (not effective/global BS)
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    )
    # Each process (GPU) gets its own copy of the model
    model = FaceNet(num_classes=len(train_dataset.data.dataset.classes), embedding_size=128, use_SE=False, ).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])  # you can also get the rank dynamically via "torch.distributed.get_rank()"
    # optim = torch.optim.SGD(model.parameters(), LR, 0.8, weight_decay=WD)
    optim = torch.optim.AdamW(model.parameters(), LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=LR, total_steps=EPOCHS * len(train_loader), pct_start=0.1, div_factor=20, final_div_factor=1e3)

    train_loss, train_acc = train(rank, EPOCHS, model, train_loader, optim, sched)

    # Since each GPU has its own copy of the same trained model, run the testing on GPU 0:
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=BS * world_size, num_workers=NUM_WORKERS, pin_memory=True)
        test_acc = test(model, test_loader)

        torch.save(
            {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'epoch': EPOCHS,
                'loss': train_loss,
                'train_acc': train_acc,
                "test_acc": test_acc,
            },
            "checkpoint.pth"
        )

    destroy_process_group()


if __name__ == "__main__":  # child processes spawned by multiprocessing do NOT execute this block! This is ONLY executed by a **single** main "parent" process.
    os.environ["MASTER_ADDR"] = "localhost"  # 127.0.0.1
    os.environ["MASTER_PORT"] = "29500"
    # v2.ColorJitter(brightness=0.2, contrast=0.2)
    T = v2.Compose([
        v2.PILToTensor(),  # dtype is torch.uint8
        v2.Resize((112, 112))
    ])
    T_train = v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3)
    ])
    T_test = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3)
    ])

    facescrub = Path("datasets/facescrub")
    dataset = ImageFolder(facescrub, transform=T)  # default image loader function is `PIL.Image.open()`

    train_data, test_data = random_split(dataset, [0.9, 0.1], generator=generator)

    class TransformData(Dataset):
        def __init__(self, data, transforms):
            self.data = data
            self.T = transforms

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x, y = self.data[idx]
            return self.T(x), y

    train_data = TransformData(train_data, T_train)
    test_data = TransformData(test_data, T_test)

    # Arguments passed to "main" child processes are pickled and sent to each. A Dataset object is often not picklable (transforms, lambdas, etc.)
    mp.spawn(main, args=(train_data, test_data), nprocs=world_size, join=True)  # first argument is implicitly the process id/index (auto-allocated as rank by DDP)
