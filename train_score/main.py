import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import argparse
import pandas as pd
from torchvision.models import ResNet34_Weights, DenseNet161_Weights, ResNeXt50_32X4D_Weights, ResNet152_Weights
import numpy as np
from model import NaiveModel, MLPModel
from utils import train_uncertainty_predictor
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--model", type=str, default="resnet34")
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
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


model = NaiveModel(nn.DataParallel(model))
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_dataset = torchvision.datasets.ImageFolder(
    root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
    transform=val_transform
)

train_size = 5000
test_size = len(test_dataset) - train_size

# Split the dataset
train_dataset, remaining_dataset = random_split(
    test_dataset,
    [train_size, test_size],
)


train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=64)

mlp = MLPModel(input_size=512, hidden_size=256).to(device)
mlp = nn.DataParallel(mlp)
mlp.train()

mlp = train_uncertainty_predictor(model, mlp, train_loader, device, args)
mlp.eval()

dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=64)

# Initialize lists to store results
all_confidences = []
all_y_hat = []
all_y_true = []

with torch.no_grad():
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        feature = model.get_feature(data)
        logits = model(data)
        prob = torch.softmax(logits, dim=-1)
        y = torch.argmax(prob, dim=-1)
        #conf = mlp(feature).view(-1)
        conf = torch.softmax(mlp(feature), dim=-1)[:, 1].view(-1)

        # Store results
        all_confidences.extend(conf.cpu().numpy())
        all_y_hat.extend(y.cpu().numpy())
        all_y_true.extend(target.cpu().numpy())


# Create DataFrame with correct labels
df = pd.DataFrame({
    'Y': all_y_true,
    'Yhat': all_y_hat,
    'confidence': all_confidences,
})

# Save to CSV
output_file = f'score_{args.model}_{args.dataset}.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Optional: Compute and print top-1 accuracy for verification
accuracy = np.mean(np.array(all_y_true) == np.array(all_y_hat))
print(f"Top-1 Accuracy: {accuracy:.4f}")