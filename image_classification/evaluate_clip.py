import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
import clip
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
imagenet_classes = []
with open('/mnt/sharedata/ssd_small/common/datasets/imagenet/class_names.txt',
          'r') as f:  # Adjust path if needed
    imagenet_classes = [line.strip() for line in f.readlines()]
# Alternatively, if you don't have a class_names.txt file, you can use the 1000-class list from torchvision
if len(imagenet_classes) != 1000:
    imagenet_classes = []
    with open('imagenet_classes.txt', 'r') as f:  # You might need to create this file or get the classes another way
        for line in f:
            imagenet_classes.append(line.strip())
    # If still not available, use a simple placeholder (model will not perform well)
    if len(imagenet_classes) != 1000:
        imagenet_classes = [f"class {i}" for i in range(1000)]

# Tokenize the text prompts (class names)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_classes]).to(device)
print(f"Loaded {len(imagenet_classes)} class names.")

# Precompute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # L2 normalize

# Use CLIP's specific preprocessing for the image
# Note: CLIP's built-in preprocess already includes Resize, CenterCrop, ToTensor, and Normalization
val_transform = preprocess

test_dataset = torchvision.datasets.ImageFolder(
    root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
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
        # This gives a [batch_size, 1000] matrix of logit scores
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # Convert logits to probabilities
        probs = logits_per_image.softmax(dim=-1)

        # Get predictions and confidences
        confidences, predictions = probs.max(dim=-1)

        # Store results
        all_confidences.extend(confidences.cpu().numpy())
        all_y_hat.extend(predictions.cpu().numpy())
        all_y_true.extend(targets.cpu().numpy())

# Create DataFrame
df = pd.DataFrame({
    'Y': all_y_true,
    'Y_hat': all_y_hat,
    'confidence': all_confidences,
})

# Save to CSV. Using the model name with slashes replaced for filename safety.
model_filename = args.model.replace('/', '_')
df.to_csv(f'CLIP_{model_filename}_{args.dataset}.csv', index=False)
print(f"Results saved to CLIP_{model_filename}_{args.dataset}.csv")