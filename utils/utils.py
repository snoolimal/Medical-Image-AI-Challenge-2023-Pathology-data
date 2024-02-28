import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
import pickle
from tqdm import tqdm
from utils.config import training_config


def get_patch_metadata(mode, risk=None, save=False):
    assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

    patch_dir = Path('dataset') / f'{mode}' / f'{mode}_patch_rs'
    patch_paths = [path for path in patch_dir.glob('*.npy')]
    patch_names = [path.stem for path in patch_dir.glob('*.npy')]
    patch_ranks = [int(name.split('_')[-1]) for name in patch_names]
    slide_names = [f'{mode}_{name.split("_")[1]}' for name in patch_names]
    patch_metadata = pd.DataFrame({
        'Patch_name': patch_names,
        'Patch_path': patch_paths,
        'Patch_rank': patch_ranks,
        'Slide_name': slide_names
    })

    slide_metadata = pd.read_csv(str(Path('dataset') / f'{mode}_dataset.csv'))
    slide_dir = Path('dataset') / f'{mode}' / f'{mode}'
    slide_metadata['Slide_path'] = [path for path in slide_dir.glob('*.png')]
    slide_metadata['Risk'] = slide_metadata['Location'].apply(lambda x: 2 if x in ['heel', 'toe', 'big toe'] else (0 if x in ['finger', 'sole'] else 1))

    patch_metadata = slide_metadata.merge(patch_metadata)
    patch_metadata.reset_index(drop=True, inplace=True)

    if risk is not None:
        patch_metadata = patch_metadata[patch_metadata['Risk'] == risk].reset_index(drop=True)

    if mode == 'train':
        patch_metadata.insert(len(patch_metadata.columns)-1, 'Recurrence', patch_metadata.pop('Recurrence'))

    if save:
        save_dir = str(Path('dataset') / f'{mode}_patch_metadata.csv')
        patch_metadata.to_csv(save_dir, index=False)
    else:
        return patch_metadata


def get_patch_stats(risk=None, save=False):
    metadata = get_patch_metadata('train', risk)
    patch_path = metadata['Patch_path'].values
    patches = [np.load(patch).astype(np.float32) for patch in tqdm(patch_path, desc='Loading Patches for Stats')]

    normalized_mean = np.mean(patches, axis=(0, 1, 2))
    normalized_mean = tuple(normalized_mean.round(3))
    normalized_std = np.std(patches, axis=(0, 1, 2))
    normalized_std = tuple(normalized_std.round(3))

    stats_dict = {'mean': normalized_mean, 'std': normalized_std}

    if save:
        with open('normalized_stats.pkl', 'wb') as f:
            pickle.dump(stats_dict, f)
    else:
        return stats_dict


def train_transforms(risk=None):
    stats = get_patch_stats(risk)

    return A.Compose([
        # A.Transpose(p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.25),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
        # A.HueSaturationValue(sat_shift_limit=20, val_shift_limit=0, p=0.5),
        # A.ChannelShuffle(p=0.5),
        A.Normalize(mean=stats['mean'], std=stats['std']),
        ToTensorV2()
    ])


def test_transforms(risk=None):
    stats = get_patch_stats(risk)

    return A.Compose([
        A.Normalize(mean=stats['mean'], std=stats['std']),
        ToTensorV2()
    ])


def set_scheduler(optimizer, training_config=training_config['unetvit']):
    config = training_config['scheduler']

    mode = 'min' if training_config['monitor'] == 'loss' else 'max'
    patience = training_config['early_stopping_rounds'] - 1
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=config['factor'],
        patience=patience,
        min_lr=config['min_lr'],
        verbose=config['verbose']
    )

    return scheduler
