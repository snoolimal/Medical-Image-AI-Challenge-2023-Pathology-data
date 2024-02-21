import numpy as np
from torch.utils.data import Dataset
from _utils import get_patch_metadata


class ABUNetDataset(Dataset):
    def __init__(self, process_type, risk=None, valid_indices=None, transforms=None):
        assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        if valid_indices is not None:
            self.metadata = get_patch_metadata('train', risk)

            if process_type == 'train':
                self.metadata = self.metadata.loc[~valid_indices, :].reset_index(drop=True)
            elif process_type == 'test':
                self.metadata = self.metadata.loc[valid_indices, :].reset_index(drop=True)
        else:
            self.metadata = get_patch_metadata('test', risk)

        self.valid_indices = valid_indices
        self.transforms = transforms

    def __getitem__(self, item):
        patch_name = self.metadata.loc[item, 'Patch_name']
        patch_path = self.metadata.loc[item, 'Patch_path']
        patch = np.load(patch_path)

        if self.transforms is not None:
            patch = self.transforms(image=patch)['image']

        if self.valid_indices is not None:
            label = self.metadata.loc[item, 'Recurrence']

            return patch, label
        else:
            return patch_name, patch

    def __len__(self):
        return len(self.metadata)


# ## Check
# import os
# import random
# from torch.utils.data import DataLoader
# length = len(os.listdir('dataset/train_patch_rs'))
# train_indices = random.choices(range(length), k=int(0.8*length))
# valid_indices = np.ones(length, dtype=bool)
# valid_indices[train_indices] = False
# dataset = ABUNetDataset('train', valid_indices)
# patch, label = dataset.__getitem__(0)
# print(patch.shape)  # (512, 512, 3)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# for patches, labels in dataloader:
#     print(patches.size(), labels.size())  # [64, 512, 512, 3], [64]
#     break
