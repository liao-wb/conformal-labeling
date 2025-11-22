import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import pandas as pd
from torchvision.models import ResNet34_Weights, DenseNet161_Weights, ResNeXt50_32X4D_Weights, ResNet152_Weights
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
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
elif model_name == "resnet152":
    model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device).eval()
elif model_name == "densenet161":
    model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1).to(device).eval()
elif model_name == "resnext50":
    model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1).to(device).eval()
else:
    raise NotImplementedError(f"Model {model_name} not supported")

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
    val_dir = "/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val"
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Use the custom dataset
    # full_ds = ImageNetV2Dataset(val_dir, transform=val_tf, classnames_file="/mnt/sharedata/ssd_small/common/datasets/imagenetv2/classnames.txt")
    test_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
        transform=val_tf
    )
    # test_dataset = torchvision.datasets.ImageFolder(
    #     root="/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val",
    #     transform=val_transform
    # )
elif args.dataset == "imagenetc1":
    val_dir = "/mnt/sharedata/ssd_small/common/datasets/imagenet-corruption/brightness/1"
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=val_tf
    )
else:
    raise NotImplementedError(f"Dataset {args.dataset} not supported")

# For ImageNetV2, remap labels to match ImageNet class indices (0-999)
# ImageFolder sorts folder names lexicographically, but folders are named '0' to '999'
label_remap = None
if args.dataset != "imagenet":
    class_names = test_dataset.classes  # Sorted list: ['0', '1', '10', ..., '999']
    label_remap = {sorted_idx: int(class_name) for sorted_idx, class_name in enumerate(class_names)}
    print(f"Applied label remapping for ImageNetV2. Mapping size: {len(label_remap)}")

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

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

        # Remap targets if needed
        batch_targets = target.cpu().numpy()
        if label_remap is not None:
            batch_targets = np.array([label_remap[t.item()] for t in target])
        all_y_true.extend(batch_targets)

# Create DataFrame with correct labels
df = pd.DataFrame({
    'Y': all_y_true,
    'Yhat': all_y_hat,
    'confidence': all_confidences,
})

# Save to CSV
output_file = f'{args.model}_{args.dataset}.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Optional: Compute and print top-1 accuracy for verification
accuracy = np.mean(np.array(all_y_true) == np.array(all_y_hat))
print(f"Top-1 Accuracy: {accuracy:.4f}")