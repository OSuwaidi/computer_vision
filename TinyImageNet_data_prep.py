# بسم الله الرحمن الرحيم و به نستعين

import shutil
from pathlib import Path

def prep_tinyimagenet_data(tiny_val_root: str) -> None:
    """
    Reorganize tiny-imagenet-200/val into ImageFolder format:
      val/<wnid>/*.JPEG
    """
    if not tiny_val_root.endswith("/val"):
        tiny_val_root += "/val"
    val_dir = Path(tiny_val_root)  # path to validation data
    img_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"

    # If it already looks like the "ImageFolder" structure (val has class dirs), do nothing.
    existing_class_dirs = [p for p in val_dir.iterdir() if p.is_dir() and p.name != "images"]
    if len(existing_class_dirs) > 10:  # heuristic
        return

    # Parse annotations, where each line: <image_name>\t<class_wnid>\t<bbox_coord>
    made = set()
    with open(ann_path, "r") as f:
        for line in f:
            img_name, wnid, *_ = line.strip().split("\t")  # "wnid" is the class, and "*_" contains bbox coordinates
            class_dir = val_dir / wnid
            if wnid not in made:
                class_dir.mkdir(parents=True, exist_ok=True)
                made.add(wnid)

            # Move images into class folders:
            src_img = img_dir / img_name
            if src_img.exists():           # idempotent
                shutil.move(src_img, class_dir)

    # optionally remove empty images dir
    if not any (img_dir.iterdir()):
        try:
            img_dir.rmdir()
            print("Removed empty 'images' directory.")
        except OSError:
            pass
    else:
        print(f"Warning: 'images' directory is not empty. Some files may have been missed.")


"""
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

T_train = v2.Compose([
    v2.RandomCrop(64, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    v2.ToTensor(),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

T_test = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tiny_root = "./data/tiny-imagenet-200"  # after extracting the zip

# One-time preparation for val split
make_tinyimagenet_val_folders(tiny_root + "/val")

train_dir = Path(tiny_root) / "train"  # already class-foldered
val_dir   = Path(tiny_root) / "val"    # now class-foldered after helper

train_ds = ImageFolder(train_dir, transform=T_train)
val_ds   = ImageFolder(val_dir,   transform=T_test)
"""
