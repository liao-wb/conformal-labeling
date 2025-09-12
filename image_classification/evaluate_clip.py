import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import clip
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model name, e.g., ViT-B/32, RN50, ViT-L/14")
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the CLIP model and the associated preprocessing
model, preprocess = clip.load(args.model, device=device)
model.eval()  # Set to evaluation mode

# Get ImageNet class names and tokenize them for CLIP
class_name = []
classnames_path = None
if args.dataset == "imagenet":
    classnames_path = '/mnt/sharedata/ssd_small/common/datasets/imagenet/classnames.txt'
elif args.dataset == "imagenetv2":
    classnames_path = '/mnt/sharedata/ssd_small/common/datasets/imagenetv2/classnames.txt'
else:
    raise NotImplementedError

try:
    with open(classnames_path, 'r') as f:
        class_name = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Warning: {classnames_path} not found. Using fallback class names.")
    class_name = [f"class {i}" for i in range(1000)]

print(f"Loaded {len(class_name)} class names.")

# Tokenize the text prompts (class names)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(device)

# Precompute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # L2 normalize

# Use CLIP's specific preprocessing for the image
val_transform = preprocess

# Get the correct IMAGE dataset path (not classnames path)
dataset_path = None
if args.dataset == "imagenet":
    dataset_path = "/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val"
elif args.dataset == "imagenetv2":
    dataset_path = "/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val"
else:
    raise NotImplementedError

print(f"Loading images from: {dataset_path}")

test_dataset = torchvision.datasets.ImageFolder(
    root=dataset_path,  # ← FIXED: Use the image path, not classnames path
    transform=val_transform
)

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Initialize lists to store results
all_confidences = []
all_y_hat = []
all_y_true = []

with torch.no_grad():
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        # Get image features
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # L2 normalize

        # Calculate similarity (cosine similarity between image and text features)
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # Convert logits to probabilities
        probs = logits_per_image.softmax(dim=-1)

        # Get predictions and confidences
        confidences, predictions = probs.max(dim=-1)

        # Store results
        all_confidences.extend(confidences.cpu().numpy())
        all_y_hat.extend(predictions.cpu().numpy())

        # Remap targets for ImageNetV2 since folders '0'-'999' are sorted alphabetically, not numerically
        if args.dataset == "imagenetv2":
            remapped_targets = [int(test_dataset.classes[t]) for t in targets.cpu().numpy()]
            all_y_true.extend(remapped_targets)
        else:
            all_y_true.extend(targets.cpu().numpy())

# Calculate accuracy for immediate feedback
accuracy = 100 * (np.array(all_y_hat) == np.array(all_y_true)).mean()
print(f"Final Accuracy: {accuracy:.2f}%")

# Create DataFrame
df = pd.DataFrame({
    'Y': all_y_true,
    'Yhat': all_y_hat,
    'confidence': all_confidences,
})

# Save to CSV
model_filename = args.model.replace('/', '_')
output_file = f'CLIP_{model_filename}_{args.dataset}.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")