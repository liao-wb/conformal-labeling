import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from utils import get_selected_dataloader, train, evaluate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', default="cbh", type=str)
args = parser.parse_args()
# -------------------------
# Paths (modify these)
# -------------------------
val_dir = "/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val"

# -------------------------
# Transforms
# -------------------------
train_tf = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# -------------------------
# Datasets & Loaders
# -------------------------
#train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
full_ds   = datasets.ImageFolder(val_dir, transform=val_tf)
train_size = int(len(full_ds) * 0.5)   # 5 000
test_size  = len(full_ds) - train_size  # 5 000

# random_split guarantees a *different* split every run
# (set generator=torch.Generator().manual_seed(42) for reproducibility)
train_ds, val_ds = random_split(
    full_ds,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(1234)   # <-- optional, fixed seed
)

print(f"Size of train set: {len(train_ds)}, Size of validation set: {len(val_ds)}")

origin_train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=32)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=32)

selected_train_loader = get_selected_dataloader(train_ds, args, alpha=0.1)

# -------------------------
# Model: Pretrained ResNet-34
# -------------------------
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
acc = evaluate(model, val_loader, device)
print(f"Before finetuning: Accuracy: {acc}")

train(model, selected_train_loader, val_loader, epochs=15, lr=3e-5)