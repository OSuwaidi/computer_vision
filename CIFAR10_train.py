# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import torch
import numpy as np
import random
from computer_vision.models.resnet import resnet20
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.auto import trange, tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

torch.cuda.empty_cache()
device = torch.device('cuda')

# Define hyperparameters:
BS = 64
LR = 0.07
EPOCHS = 101
NUM_WORKERS = 0

model = resnet20(3, 10).to(device)

norm = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
train_T = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25), transforms.RandomCrop(32, padding=4), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), norm])
test_T = transforms.Compose([transforms.ToTensor(), norm])

train_data = datasets.CIFAR10(root='datasets/CIFAR10', train=True, transform=train_T, download=True)
test_data = datasets.CIFAR10(root='datasets/CIFAR10', train=False, transform=test_T)

# If you set "num_workers" > 0, each worker in "DataLoader" will have its own RNG state, independent on the global RNG
train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, generator=torch.Generator().manual_seed(seed))
test_loader = DataLoader(test_data, batch_size=10_000, pin_memory=True, num_workers=NUM_WORKERS)

# Note: the "lr" value you use in "optim" doesn't matter since it depends on the value you input into the "scheduler"
optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # As weight_decay increases, more weight penalization (more sensitive to large weights)
# Therefore, harder/slower to achieve convergence, but more robust against overfitting (more generalizable).
# As batch size decreases, weight decay should also decrease (need more room/flexibility when dealing with noisy updates)
# As model gets more complex (more parameters) increasing the weight decay hinders the learning process, thus reducing overall performance and vice-versa!
# As model gets less complex (fewer parameters), it's easier for the model to fit the training data (even though not as good of a fit)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))

losses = []
for e in trange(EPOCHS, desc="Training Epochs", leave=True):
    epoch_loss = 0
    for x, y in tqdm(train_loader, leave=False, desc=f"Epoch {e+1}/{EPOCHS} Progress", position=0):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward pass and compute loss
        loss = F.cross_entropy(model(x), y)
        epoch_loss += loss.item()

        # Backward pass and optimizer/scheduler step
        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step()  # OneCycleLR works on a per-batch basis

    # Log the average loss for each epoch
    losses.append(epoch_loss / len(train_loader))

model.eval()
accuracy = 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        predictions = model(x).argmax(1)
        accuracy += (predictions == y).float().sum().item()

print(f'Accuracy: {accuracy/len(test_data)*100:.4}%')

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Default Training Loss")
plt.savefig("training_loss.png")
