import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.data import random_split
import argparse
import pandas as pd
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet152(pretrained=True).to(device).eval()

# Your transforms and dataset setup
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cal_test_dataset = torchvision.datasets.ImageFolder(
    root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
    transform=val_transform
)

cal_dataset, test_dataset = random_split(cal_test_dataset, [int(0.2 * len(cal_test_dataset)), len(cal_test_dataset) - int(0.2 * len(cal_test_dataset))])

cal_dataloader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

T = torch.tensor(1.0, dtype=torch.float, device=device, requires_grad=True)
optimizer = torch.optim.SGD([T], lr=0.01)  # Note: T should be in a list

model.train()
for epoch in range(20):
    total_loss = 0.0
    for data, target in cal_dataloader:
        data, target = data.to(device), target.to(device)

        # Forward pass through model
        with torch.no_grad():  # Don't update model weights
            logits = model(data)

        # Apply temperature scaling
        scaled_logits = logits / T

        # Calculate cross entropy loss
        loss = F.cross_entropy(scaled_logits, target)

        # Backward pass and optimize only T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()



# Initialize lists to store results
confidences = []
calibrated_confidences = []
y_hat = []
y_true = []



with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)

        logits = model(data)

        prob = torch.softmax(logits, dim=-1)
        calibrated_prob = F.softmax(logits / T, dim=-1)

        y_1 = torch.argmax(prob, dim=-1)

        conf = prob[torch.arange(prob.size(0)), y_1]
        calibrated_conf = calibrated_prob[torch.arange(calibrated_prob.size(0)), y_1]

        # Store results
        confidences.extend(conf.cpu().numpy())
        calibrated_confidences.extend(calibrated_conf.cpu().numpy())

        y_hat.extend(y_1.cpu().numpy())
        y_true.extend(target.cpu().numpy())

# Create DataFrame
df = pd.DataFrame({
    'Y': y_true,
    'Y_hat_resnet34': y_hat,
    'confidence': confidences,
    'calibrated_confidence': calibrated_confidences,
})

# Save to CSV
df.to_csv('model_predictions.csv', index=False)