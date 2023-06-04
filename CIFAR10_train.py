# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import torch
from torchvision import datasets, transforms
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from resnet import resnet20

device = torch.device('cuda')
torch.cuda.empty_cache()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_T = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), norm])
test_T = transforms.Compose([transforms.ToTensor(), norm])

train_data = datasets.CIFAR10(root='datasets/CIFAR10', train=True, transform=train_T, download=True)
test_data = datasets.CIFAR10(root='datasets/CIFAR10', train=False, transform=test_T, download=True)

BS = 100
LR = 0.3
EPOCHS = 10

train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1000, pin_memory=True)

m = resnet20(32).to(device)
# Note: the "lr" value you use in "optim" doesn't matter since it depends on the value you input into the "scheduler"
optim = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # As weight_decay increases, more weight penalization (more sensitive to large weights)
# Therefore, harder/slower to achieve convergence, but more robust against overfitting (more generalizable).
# As batch size decreases, weight decay should also decrease (need more room/flexibility when dealing with noisy updates)
# As model gets more complex (more parameters) increasing the weight decay hinders the learning process, thus reducing overall performance and vice-versa!
# As model gets less complex (fewer parameters), it's easier for the model to fit the training data (even though not as good of a fit)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))

for e in trange(EPOCHS):
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()

m.eval()
accuracy = []
with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        _, preds = m(x).max(1)
        accuracy.append((preds == y).float().mean().item())

print(f'Accuracy: {np.mean(accuracy)*100:.4}%')
