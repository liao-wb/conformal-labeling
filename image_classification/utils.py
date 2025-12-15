"""
Utility functions for Image Classification experiments.
Handles model loading, dataset preparation, and common transforms.
"""

import os
from typing import Tuple, Dict, Optional, Callable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import (
    ResNet34_Weights, 
    ResNet152_Weights, 
    DenseNet161_Weights, 
    ResNeXt50_32X4D_Weights
)

# Constants for ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform(resize: int = 256, crop: int = 224) -> transforms.Compose:
    """Standard ImageNet validation transform."""
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Loads a pre-trained model and moves it to the device."""
    model_name = model_name.lower()
    
    if model_name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
    elif model_name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = models.resnet152(weights=weights)
    elif model_name == "densenet161":
        weights = DenseNet161_Weights.IMAGENET1K_V1
        model = models.densenet161(weights=weights)
    elif model_name == "resnext50":
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        model = models.resnext50_32x4d(weights=weights)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model.to(device).eval()

def get_dataset(
    dataset_name: str, 
    root_dir: str, 
    transform: Optional[Callable] = None
) -> Tuple[torchvision.datasets.ImageFolder, Optional[Dict[int, int]]]:
    """
    Loads the dataset and returns it along with an optional label remapping.
    
    Args:
        dataset_name: Name of the dataset (imagenet, imagenetv2, imagenet-c, etc.)
        root_dir: Base directory containing the datasets.
        transform: Transformations to apply.
        
    Returns:
        dataset: The PyTorch dataset.
        label_remap: Dictionary mapping current labels to ImageNet indices (if needed).
    """
    if transform is None:
        transform = get_transform()

    label_remap = None
    dataset_path = ""

    if dataset_name == "imagenet":
        dataset_path = os.path.join(root_dir, "imagenet/val")
    elif dataset_name == "imagenetv2":
        dataset_path = os.path.join(root_dir, "imagenetv2/imagenetv2-matched-frequency-format-val")
    elif dataset_name.startswith("imagenetc"):
        # Expecting format "imagenetcX" where X is severity 1-5
        severity = dataset_name[-1]
        dataset_path = os.path.join(root_dir, f"imagenet-corruption/brightness/{severity}")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

    # ImageNet-V2 remapping logic
    if dataset_name == "imagenetv2":
        # Folder names are '0', '1', ... '999'. 
        # ImageFolder sorts them alphabetically ('0', '1', '10', '100'...), so we remap.
        class_names = dataset.classes
        label_remap = {sorted_idx: int(class_name) for sorted_idx, class_name in enumerate(class_names)}
        print(f"Applied label remapping for ImageNetV2. Mapping size: {len(label_remap)}")

    return dataset, label_remap

