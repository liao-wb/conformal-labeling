import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageNet
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--classname', type=str, default='goldfish')
args = parser.parse_args()
# Load ImageNet (validation set by default)
dataset = ImageNet(root='/mnt/sharedata/ssd_small/common/datasets/imagenet', split='val')

# Get class names
class_names = dataset.classes

# Function to show an image
def show_single_image(image, label, class_name):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Class: {class_name}\nLabel: {label}")
    plt.axis('off')
    plt.show()

# --- Specify the desired class (e.g., "goldfish") ---
desired_class = args.classname  # Change this to any class in ImageNet

# Find all indices of images belonging to the desired class
class_idx = class_names.index(desired_class)  # Get the class index
indices_of_class = [i for i, (_, label) in enumerate(dataset) if label == class_idx]

if not indices_of_class:
    print(f"No images found for class: {desired_class}")
    exit()

# Randomly select one image from this class
random_idx = np.random.choice(indices_of_class)
image, label = dataset[random_idx]

# Display the image
show_single_image(image, label, class_names[label])

# Save the image (avoid overwriting existing files)
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)  # Create dir if it doesn't exist

i = 0
while True:
    path = os.path.join(save_dir, f"{class_names[label]}_{i}.jpg")
    if not os.path.exists(path):
        break
    i += 1

image.save(path)
print(f"Saved to: {path}")

print("\nClass Name:", class_names[label])