import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import pandas as pd
import numpy as np
import json
from torchvision.models import ResNet34_Weights, DenseNet161_Weights, ResNeXt50_32X4D_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "imagenetv2"])
parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34", "densenet161", "resnext50"])
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Load model -----
if args.model == "resnet34":
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = models.resnet34(weights=weights).to(device).eval()
elif args.model == "densenet161":
    weights = DenseNet161_Weights.IMAGENET1K_V1
    model = models.densenet161(weights=weights).to(device).eval()
elif args.model == "resnext50":
    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    model = models.resnext50_32x4d(weights=weights).to(device).eval()
else:
    raise NotImplementedError

# Get the ImageNet-1k class categories (ordered list of 1000 class labels)
imagenet_classes = weights.meta["categories"]

# ----- Transforms -----
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----- Dataset -----
if args.dataset == "imagenet":
    test_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
        transform=val_transform
    )

elif args.dataset == "imagenetv2":
    test_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val",
        transform=val_transform
    )

    # Fix the class_to_idx mapping (make it match original ImageNet order)
    # test_dataset.classes = list of wnids (e.g., n01440764, n01443537, ...)
    wnid_to_idx = {wnid: i for i, wnid in enumerate(imagenet_classes)}
    test_dataset.class_to_idx = wnid_to_idx
    test_dataset.targets = [wnid_to_idx[test_dataset.classes[label]] for label in test_dataset.targets]

else:
    raise NotImplementedError

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ----- Evaluation -----
all_confidences = []
all_y_hat = []
all_y_true = []

with torch.no_grad():
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        logits = model(data)
        prob = torch.softmax(logits, dim=-1)

        y = torch.argmax(prob, dim=-1)
        conf = prob[torch.arange(prob.size(0)), y]

        all_confidences.extend(conf.cpu().numpy())
        all_y_hat.extend(y.cpu().numpy())
        all_y_true.extend(target.cpu().numpy())

# ----- Save results -----
df = pd.DataFrame({
    'Y': all_y_true,
    'Yhat': all_y_hat,
    'confidence': all_confidences,
})
df.to_csv(f'{args.model}_{args.dataset}.csv', index=False)

# Print quick accuracy check
acc = np.mean(np.array(all_y_hat) == np.array(all_y_true))
print(f"Top-1 Accuracy on {args.dataset}: {acc * 100:.2f}%")
