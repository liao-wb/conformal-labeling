import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--model", type=str, default="resnet34")
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = args.model

if model_name == "resnet34":
    model = models.resnet34(pretrained=True).to(device).eval()
elif model_name == "densenet161":
    model = models.densenet161(pretrained=True).to(device).eval()
elif model_name == "resnext50":
    model = models.resnext50_32x4d(pretrained=True).to(device).eval()
else:
    raise NotImplementedError


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
else: raise NotImplementedError

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize lists to store results
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

        # Store results
        all_confidences.extend(conf.cpu().numpy())

        all_y_hat.extend(y.cpu().numpy())

        all_y_true.extend(target.cpu().numpy())

# Create DataFrame
df = pd.DataFrame({
    'Y': all_y_true,
    'Yhat': all_y_hat,
    'confidence': all_confidences,

})

# Save to CSV
df.to_csv(f'{args.model}_{args.dataset}.csv', index=False)