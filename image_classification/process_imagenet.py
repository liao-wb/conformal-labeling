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
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = models.resnet34(pretrained=True).to(device).eval()
model2 = models.resnet50(pretrained=True).to(device).eval()
model3 = models.resnet152(pretrained=True).to(device).eval()

# Your transforms and dataset setup
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = torchvision.datasets.ImageFolder(
    root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/val",
    transform=val_transform
)

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize lists to store results
all_confidences_1 = []
all_confidences_2 = []
all_confidences_3 = []
all_y_hat_1 = []
all_y_hat_2 = []
all_y_hat_3 = []
all_y_true = []

with torch.no_grad():
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        logits_1 = model1(data)
        logits_2 = model2(data)
        logits_3 = model3(data)

        prob_1 = torch.softmax(logits_1, dim=-1)
        prob_2 = torch.softmax(logits_2, dim=-1)
        prob_3 = torch.softmax(logits_3, dim=-1)

        y_1 = torch.argmax(prob_1, dim=-1)
        y_2 = torch.argmax(prob_2, dim=-1)
        y_3 = torch.argmax(prob_3, dim=-1)

        conf_1 = prob_1[torch.arange(prob_1.size(0)), y_1]
        conf_2 = prob_2[torch.arange(prob_2.size(0)), y_2]
        conf_3 = prob_3[torch.arange(prob_3.size(0)), y_3]

        # Store results
        all_confidences_1.extend(conf_1.cpu().numpy())
        all_confidences_2.extend(conf_2.cpu().numpy())
        all_confidences_3.extend(conf_3.cpu().numpy())
        all_y_hat_1.extend(y_1.cpu().numpy())
        all_y_hat_2.extend(y_2.cpu().numpy())
        all_y_hat_3.extend(y_3.cpu().numpy())
        all_y_true.extend(target.cpu().numpy())

# Create DataFrame
df = pd.DataFrame({
    'Y': all_y_true,
    'Y_hat_resnet34': all_y_hat_1,
    'Y_hat_resnet50': all_y_hat_2,
    'Y_hat_resnet152': all_y_hat_3,
    'confidence_resnet34': all_confidences_1,
    'confidence_resnet50': all_confidences_2,
    'confidence_resnet152': all_confidences_3
})

# Save to CSV
df.to_csv('model_predictions.csv', index=False)