import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from .utils import get_selected_dataloader, train, evaluate
from .custom_dataset import ImageNetV2Dataset
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

if args.dataset == "imagenetv2":
    val_dir = "/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val"
    # Use the custom dataset
    test_dataset = ImageNetV2Dataset(val_dir, transform=val_tf, classnames_file="/mnt/sharedata/ssd_small/common/datasets/imagenetv2/classnames.txt")
elif args.dataset == "imagenet":
    test_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
        transform=val_tf
    )
else:
    raise NotImplementedError
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).eval().to("cuda")

test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=64)
