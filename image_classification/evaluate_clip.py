"""
CLIP Model Evaluation Script.
Performs zero-shot classification using OpenAI's CLIP models.
"""

import argparse
import os
import sys

import clip
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot Evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing datasets")
    parser.add_argument("--classnames_file", type=str, default="classnames.txt", 
                        help="Path to file containing class names")
    parser.add_argument("--model", type=str, default="ViT-B/32", 
                        help="CLIP model architecture")
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()

def load_class_names(filepath: str) -> list:
    """Loads class names from a text file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Generating dummy class names.")
        return [f"class {i}" for i in range(1000)]
    
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_clip_dataset(dataset_name, root_dir, transform):
    """Specific dataset loader for CLIP (handles structure similar to utils.py)."""
    dataset_path = ""
    if dataset_name == "imagenet":
        dataset_path = os.path.join(root_dir, "imagenet/val")
    elif dataset_name == "imagenetv2":
        dataset_path = os.path.join(root_dir, "imagenetv2/imagenetv2-matched-frequency-format-val")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for CLIP evaluation.")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
    return torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CLIP
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    # Prepare Text Features (Classifier)
    class_names = load_class_names(args.classnames_file)
    print(f"Loaded {len(class_names)} class names.")
    
    text_prompts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Prepare Data
    dataset = get_clip_dataset(args.dataset, args.data_root, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)

    # Inference Loop
    all_preds, all_confs, all_targets = [], [], []

    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        
        for images, targets in tqdm(dataloader, desc="CLIP Inference"):
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * image_features @ text_features.t()
            probs = logits.softmax(dim=-1)
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

            # ImageNet-V2 remapping logic
            if args.dataset == "imagenetv2":
                remapped_targets = [int(dataset.classes[t]) for t in targets.numpy()]
                all_targets.extend(remapped_targets)
            else:
                all_targets.extend(targets.numpy())

    # Save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.DataFrame({
        'Y': all_targets,
        'Yhat': all_preds,
        'confidence': all_confs,
    })
    
    safe_model_name = args.model.replace('/', '_')
    output_file = os.path.join(args.output_dir, f'CLIP_{safe_model_name}_{args.dataset}.csv')
    df.to_csv(output_file, index=False)
    
    acc = np.mean(np.array(all_targets) == np.array(all_preds))
    print(f"Saved: {output_file} | Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
