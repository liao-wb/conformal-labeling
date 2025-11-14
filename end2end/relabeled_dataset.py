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
        label = self.relabel[idx]

        return image, label