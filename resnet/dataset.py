import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils.utils import get_patch_metadata


class RNDataset(Dataset):
    def __init__(self, risk, mode):
        assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

        self.mode = mode
        self.metadata = get_patch_metadata(mode, risk)

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

    def __getitem__(self, item):
        patch_name = self.metadata.loc[item, 'Patch_name']
        patch_path = self.metadata.loc[item, 'Patch_path']
        patch = np.load(patch_path)
        patch = self.transforms(patch)

        if self.mode == 'train':
            patch = self.augmentations(patch)

            label = self.metadata.loc[item, 'Recurrence']
            label = torch.tensor(label)

            return patch, label
        elif self.mode == 'test':
            return patch_name, patch

    def __len__(self):
        return len(self.metadata)
