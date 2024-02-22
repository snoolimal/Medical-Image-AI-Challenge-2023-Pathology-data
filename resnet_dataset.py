import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from _utils import get_patch_metadata


class ResNetDataset(Dataset):
    def __init__(self, risk, process_type):
        assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        self.process_type = process_type
        self.metadata = get_patch_metadata(self.process_type, risk)

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

        if self.process_type == 'train':
            patch = self.augmentations(patch)

            label = self.metadata.loc[item, 'Recurrence']
            label = torch.tensor(label)

            return patch, label
        elif self.process_type == 'test':
            return patch_name, patch

    def __len__(self):
        return len(self.metadata)
