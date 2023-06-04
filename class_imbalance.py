# بسم الله الرحمن الرحيم و به نستعين

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = (15, 6)  # Length, Width
plt.style.use('seaborn')
torch.seed()
torch.manual_seed(0)
np.random.seed(0)

T = transforms.ToTensor()
dataset = datasets.FashionMNIST('datasets/FashionMNIST', transform=T)
targets = dataset.targets
classes, class_counts = np.unique(targets, return_counts=True)
nb_classes = len(classes)

# Create artificial imbalanced class counts
imbal_class_counts = [500, 5000] * 5

# Get class indices
class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
print(class_indices)

# Get imbalanced number of instances
imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
imbal_class_indices = np.hstack(imbal_class_indices)

# Set target and data to dataset
dataset.targets = targets[imbal_class_indices]
dataset.data = dataset.data[imbal_class_indices]

loader = DataLoader(dataset, batch_size=100)
hist1 = np.zeros(10)
# Here we have an imbalanced dataset
for data, target in loader:
    cls, count = np.unique(target.numpy(), return_counts=True)
    hist1[cls] += count

plt.subplot(121)
plt.title('Imbalanced')
plt.xticks(range(0, 10))
plt.bar(range(0, 10), hist1, color='g')

targets = dataset.targets
class_count = np.unique(targets, return_counts=True)[1]

weight = 1 / class_count  # len(weight) = 10
samples_weight = weight[targets]  # Will assign the value of the 1st index in "weight" (weight[0]) with everywhere a 0 exists in "targets", and will assign the value of the 2nd index in "weight" (weight[1]) with everywhere a 1 exists in "targets" and so on
w_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))  # len(samples_weight) = len(targets)

weighted_loader = DataLoader(dataset, batch_size=100, sampler=w_sampler)

# Here we have a balanced dataset due to oversampling
hist2 = np.zeros(10)
# Here we have an imbalanced dataset
for data, target in weighted_loader:
    cls, count = np.unique(target.numpy(), return_counts=True)
    hist2[cls] += count

plt.subplot(122)
plt.title('Balanced')
plt.xticks(range(0, 10))
plt.bar(range(0, 10), hist2, color='r')
plt.show()
