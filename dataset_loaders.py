import torchvision
from torchvision import transforms

# 1. Define Transformations
# We need separate transforms for Training (with augmentation) and Testing (clean).
# normalization stats below are standard ImageNet means/stds.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),        # Resize first to a slightly larger square
    transforms.RandomCrop(224),           # Random crop to target size (Data Augmentation)
    transforms.RandomHorizontalFlip(),    # Flip horizontally (Data Augmentation)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),                # Convert to Tensor (0-1 range)
    transforms.Normalize(                 # Normalize to standard distribution
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),        # deterministic resize for testing
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# --- STL10 ---
# STL10 is unique; it has 'train', 'test', and 'unlabeled' splits
stl10_data = torchvision.datasets.STL10(
    root='./data', 
    split='train',       # Options: 'train', 'test', 'unlabeled', 'train+unlabeled'
    download=True, 
    transform=train_transforms
)

# --- FGVC Aircraft ---
# FGVC has hierarchy: 'variant' (hardest), 'family' (medium), 'manufacturer' (easiest)
aircraft_data = torchvision.datasets.FGVCAircraft(
    root='./data', 
    split='train',       # Options: 'train', 'val', 'trainval', 'test'
    annotation_level='family',
    download=True, 
    transform=train_transforms
)

# --- Caltech 101 ---
# requires manual train/test split
caltech_data = torchvision.datasets.Caltech101(
    root='./data',
    target_type='category',
    download=True,
    transform=train_transforms
)

# EuroSAT
# requires manual train/test split
eurosat_full = torchvision.datasets.EuroSAT(root="./data", download=True, transform=None)


# Corrputed CIFAR10:
"$pip install autoattack robustbench"
from robustbench.data import load_cifar10c

# Load CIFAR-10-C (Corrupted)
# severities: 1 (light) to 5 (severe)
# corruptions: 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', etc.
x_test, y_test = load_cifar10c(n_examples=1000, corruptions=['snow'], severity=5)

print(f"Loaded {x_test.shape} corrupted images.")


# Train on MNIST, test on SVHN:
from torchvision import datasets, transforms

# Source: MNIST (Black and white handwritten digits)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=32)

# Target: SVHN (Color real-world house numbers) - Radical Domain Shift
ood_loader = torch.utils.data.DataLoader(
    datasets.SVHN('data', split='test', download=True, transform=transforms.Compose([
        transforms.Grayscale(), # Force 1 channel to match MNIST
        transforms.Resize((28, 28)), # Resize to match MNIST
        transforms.ToTensor()
    ])),
    batch_size=32)


# WILDS datasets:
"$pip install wilds"

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

# Load Camelyon17 (Medical Domain Shift)
dataset = get_dataset(dataset="camelyon17", download=True)

# Get the training set (Source Domain)
train_data = dataset.get_subset("train", transform=None)

# Get the OOD test set (Target Domain - unseen hospitals)
# WILDS splits are specifically designed for OOD testing (e.g., 'id_test' vs 'test')
ood_test_data = dataset.get_subset("test", transform=None)

# Create loader
train_loader = get_eval_loader("standard", train_data, batch_size=16)