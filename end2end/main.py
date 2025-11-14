import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from .utils import get_selected_dataloader, train, evaluate
from .custom_dataset import ImageNetV2Dataset
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', default="cbh", type=str)
args = parser.parse_args()

val_dir = "/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val"
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
# Use the custom dataset
#full_ds = ImageNetV2Dataset(val_dir, transform=val_tf, classnames_file="/mnt/sharedata/ssd_small/common/datasets/imagenetv2/classnames.txt")
full_ds = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
        transform=val_tf
    )
train_size = int(len(full_ds) * 0.5)   # 5 000
test_size  = len(full_ds) - train_size  # 5 000

# random_split guarantees a *different* split every run
# (set generator=torch.Generator().manual_seed(42) for reproducibility)
train_ds, val_ds = random_split(
    full_ds,
    [train_size, test_size]
)

print(f"Size of train set: {len(train_ds)}, Size of validation set: {len(val_ds)}")

origin_train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=32)
val_loader   = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=32)

selected_train_loader = get_selected_dataloader(train_ds, args, alpha=0.1)

# -------------------------
# Model: Pretrained ResNet-34
# -------------------------
#model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model = models.resnet18()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
acc = evaluate(model, val_loader, device)
print(f"Before finetuning: Val Accuracy: {acc}")

train(model, selected_train_loader, val_loader, epochs=100, lr=1e-3, weight_decay=1e-4)