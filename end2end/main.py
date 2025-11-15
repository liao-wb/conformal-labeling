import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from .utils import get_selected_dataloader, train, evaluate
from .relabeled_dataset import RemapDataset
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', default="cbh", type=str)
parser.add_argument('--dataset', default="imagenetv2", type=str)
args = parser.parse_args()


val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
# Use the custom dataset

if args.dataset == "imagenet":
    full_ds = torchvision.datasets.ImageFolder(
            root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
            transform=val_tf
        )
elif args.dataset == "imagenetv2":
    val_dir = "/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val"
    full_ds = torchvision.datasets.ImageFolder(root=val_dir,
            transform=val_tf)

    class_names = full_ds.classes  # Sorted list: ['0', '1', '10', ..., '999']
    label_remap = {sorted_idx: int(class_name) for sorted_idx, class_name in enumerate(class_names)}
    print(f"Applied label remapping for ImageNetV2. Mapping size: {len(label_remap)}")
    full_ds = RemapDataset(full_ds, label_remap)

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
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
acc = evaluate(model, val_loader, device)
print(f"Before finetuning: Val Accuracy: {acc}")

train(model, selected_train_loader, val_loader, epochs=50, lr=1e-4, weight_decay=1e-3)