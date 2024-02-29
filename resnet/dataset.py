import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class RNDataset(Dataset):
    def __init__(self, metadata, augmentations=None):
        self.metadata = metadata
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.augmentations = augmentations

    def __getitem__(self, item):
        patch_path = self.metadata.loc[item, 'Patch_path']
        patch = np.load(patch_path)
        patch = self.transforms(patch).float()

        if 'Recurrence' in self.metadata.columns:
            label = self.metadata.loc[item, 'Recurrence']
            label = torch.tensor(label, dtype=torch.float32)

            if self.augmentations is not None:
                patch = self.augmentations(patch)

            return patch, label
        else:
            patch_name = self.metadata.loc[item, 'Patch_name']

            return patch_name, patch

    def __len__(self):
        return len(self.metadata)
