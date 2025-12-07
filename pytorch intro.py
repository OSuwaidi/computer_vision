# بسم الله الرحمن الرحيم

import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# number of subprocesses to use for data loading

# how many samples per batch to load
batch_size = 20
device = torch.device('cuda')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)  # default is "train=True"
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# prepare data loaders
train_loader = DataLoader(train_data, batch_size=batch_size)  # DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, timeout=0, prefetch_factor=2, persistent_workers=False)
test_loader = DataLoader(test_data, batch_size=batch_size)
# "num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process --> If num_workers=2 you have at most 2 workers simultaneously pushing data (batches) into RAM
# "num_workers" is analogous with number of CPU cores available to load/fetch data
# "pin_memory": automatically puts the fetched/read data Tensors into pinned memory (memory that is persistent, non-paged memory) for Direct GPU (Memory) Access (DMA).
# It removes the need for: fetch data from CPU pageable memory -> driver (CUDA) copies it to temporary pinned buffer -> GPU direct memory access -> free buffer
# That gets streamlined with "pin_memory" into: fetch data from CPU pinned (reserved) memory (directly) ->  GPU direct memory access
# "drop_last": drops the last non-full batch of each worker’s dataset replica --> Used when (number_of_images/batch_size) doesn't yield a whole number
# "prefetch_factor": number of samples loaded in advance by each worker. 2 means there will be a total of 2 * num_workers samples prefetched across all workers
# "persistent_workers": if True, the data loader will not shutdown the worker processes after a dataset has been consumed once


# obtain one batch of training images:
print(train_data, "\n")  # Since we have 60000 samples and we divided them into 20 samples each batch -> our number of batches is 3000 (each of size 20)
train_iter = iter(train_loader)
images, labels = next(train_iter)  # Returns a pair; first element = collection of images stacked in columns, second element = a vector of corresponding labels from 0 to 9
print(images.shape)
images = images.numpy()  # Done such that we can plot them using matplotlib


# plot the images in the batch, along with the corresponding labels:
plt.figure(figsize=(25, 4))  # (length, height)
for i in range(len(images)):
    plt.subplot(2, 10, i+1)
    plt.title(labels[i].item())  # Used ".item" since "labels" contains values stored in tensors: "tensor(#)"
    plt.imshow(np.squeeze(images[i]), cmap='gray')  # "np.squeeze()" removes single-dimensional entries from the shape of an array
    plt.xticks([])
    plt.yticks([])
plt.show()

# View an image in more detail:
img = np.squeeze(images[1])
plt.figure(figsize=(12, 8))  # (length, height) --> (column, row)
plt.imshow(img, cmap='gray')
r, c = img.shape
thresh = img.max() / 2.5
for x in range(r):
    for y in range(c):
        val = round(img[x, y], 2) if img[x, y] != 0 else 0
        plt.annotate(val, xy=(y, x),
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='white' if img[x, y] < thresh else 'black')  # Annotates every single pixel in "img" with it's intensity value
plt.show()


# Define the Network Architecture:
# The architecture will be responsible for taking a 784-dim Tensor as input of pixel values for each image, and producing a Tensor of length 10 (our number of classes) that indicates the class scores for an input image.
# This particular example uses two hidden layers and a dropout to avoid overfitting
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)  # Takes entire image pixels as inputs (28x28)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.drop = nn.Dropout(p=0.2)  # Regularization technique to prevent overfitting

    def forward(self, x):
        x = x.view(batch_size, 784)  # "x.view(#inputs/images, image_size (flattened))"
        x = F.relu(self.drop(self.fc1(x)))  # *NOTE*: Number of rows = Number of features
        x = F.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return x


model = Net()
model.to(device)
print(model)


# Specify Loss Function and Optimizer:
# It's recommended that you use cross-entropy loss for classification problems.
# PyTorch's cross entropy function applies a softmax function to the output layer and then calculates the log loss.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []
for i in range(1):
    cost = 0
    count = -1
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        count += 1
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  # Softmax requires the prediction output towards each class (therefore 10 outputs/columns per sample in our case)
        with torch.no_grad():
            cost += loss.item()
        loss.backward()
        optimizer.step()  # Gets executed 3000 times per epoch
        if not count % 100:
            print(f'The model is training...{".."*int(count/45)}')
    print(f"Completed {i+1} Epoch(s)! \n")
    losses.append(cost/len(train_loader))

# Visualizing the training loss averaged for each epoch:
plt.plot(losses, c='μ')
plt.grid()
plt.title('Mini-Batch Gradient Descent')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.show()


# Evaluating the model:
model.eval()  # Turns off "dropout"
pred_class = np.zeros(10)
act_class = np.zeros(10)
with torch.no_grad():  # Avoid tracking for gradient to save some computation time
    for images, target in test_loader:  # We're taking 20 pairs of images and their corresponding labels for each parameter update
        images, target = images.to(device), target.to(device)
        # Forward pass: compute predicted outputs by passing inputs into the model
        outputs = model(images)
        # We calculate the probabilities; this is done by converting the output probabilities to predicted the class
        _, predictions = outputs.max(1)  # Find the maximum element *along* every row (across columns) (20x10 --> 20x1)
        # Comparing the predictions to the actual labels in the given dataset:
        for i in range(len(target)):  # Loops over all actual labels "target" for the 20 images
            if predictions[i] == target[i]:
                pred_class[predictions[i]] += 1
            act_class[target[i]] += 1


for i in range(10):
    accuracy = (pred_class[i] / act_class[i]) * 100
    print(f"The accuracy of testing class {i} is {accuracy:.4}")

# The overall testing accuracy:
avg_accuracy = (np.sum(pred_class) / np.sum(act_class)) * 100
print(f"\nThe average accuracy of model is {avg_accuracy:.4}\n")


# Visualize Sample Test Results:
print(test_data)
test_iter = iter(test_loader)
images, labels = next(test_iter)

outputs = model(images.to(device))
_, predictions = outputs.max(1)  # "tensor.max(*the dimension to reduce*)" 1 ==> column dimension
plt.figure(figsize=(25, 4))
for i in range(len(images)):
    plt.subplot(2, 10, i+1)
    plt.title(f"{int(predictions[i])} ({labels[i].item()})", c='g' if predictions[i] == labels[i] else 'red')
    plt.imshow(np.squeeze(images[i]), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
