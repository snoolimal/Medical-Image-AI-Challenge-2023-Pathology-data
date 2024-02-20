import numpy as np
import torch
from torchvision.transforms import transforms
from _utils import get_patch_metadata


class ResNetDataset:
    def __init__(self, risk, process_type):
        assert 'process_type' in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        self.process_type = process_type
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

        self.metadata = get_patch_metadata(self.process_type)
        self.metadata = self.metadata[self.metadata['Risk'] == risk]

    def __getitem__(self, item):
        patch = np.load(self.metadata.loc[item, 'Patch_path'])
        patch = self.transforms(patch)

        if self.process_type == 'train':
            patch = self.augmentations(patch)
            label = self.metadata.loc[item, 'Recurrence']
            label = torch.tensor(label)

            return patch, label
        elif self.process_type == 'test':
            return patch

    def __len__(self):
        return len(self.metadata)
