import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils.utils import get_patch_metadata


class RNDataset(Dataset):
    def __init__(self, risk, mode, valid_indices=None, augmentations=None):
        assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

        self.mode = mode
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.augmentations = augmentations

        if valid_indices is not None:
            self.metadata = get_patch_metadata('train', risk)

            if mode == 'train':
                self.metadata = self.metadata.loc[~valid_indices, :].reset_index(drop=True)
            elif mode == 'test':
                self.metadata = self.metadata.loc[valid_indices, :].reset_index(drop=True)
        else:
            self.metadata = get_patch_metadata('test', risk)

        self.valid_indices = valid_indices

    def __getitem__(self, item):
        patch_path = self.metadata.loc[item, 'Patch_path']
        patch = np.load(patch_path)
        patch = self.transforms(patch).float()

        if self.augmentations is not None:
            patch = self.augmentations(patch)

        if self.valid_indices is not None:
            label = self.metadata.loc[item, 'Recurrence']
            label = torch.tensor(label, dtype=torch.float32)

            return patch, label
        else:
            patch_name = self.metadata.loc[item, 'Patch_name']

            return patch_name, patch

    def __len__(self):
        return len(self.metadata)
