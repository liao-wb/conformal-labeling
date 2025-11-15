from torch.utils.data import Dataset


class RelabeledDataset(Dataset):
    def __init__(self, selected_dataset, relabel):
        """
        Args:
            original_dataset: The original ImageFolder dataset
            label_mapping: Dict mapping old_label -> new_label
            specific_indices: Dict mapping indices -> new_labels for specific samples
        """
        self.relabel = relabel
        self.original_dataset = selected_dataset
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        label = self.relabel[idx].item()

        return image, label


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
# 假设你的 folder 是 0~999，但顺序与 classnames.txt 一致
# 即：folder 0 → classnames.txt 第 0 行
# folder 1 → 第 1 行
# end2end/custom_dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image

# custom_dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# custom_dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageNetV2Dataset(Dataset):
    def __init__(self, root_dir, classnames_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 1. 读取 classnames.txt（确保 1000 行）
        with open(classnames_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1000, f"Expected 1000 classes, got {len(lines)}"

        # 2. folder '0' → label 0, '1' → 1, ...
        self.folder_to_idx = {str(i): i for i in range(1000)}

        # 3. 构建样本
        self.samples = []
        for folder in sorted(os.listdir(root_dir), key=int):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            idx = self.folder_to_idx.get(folder)
            if idx is None:
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, torch.tensor(idx)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label