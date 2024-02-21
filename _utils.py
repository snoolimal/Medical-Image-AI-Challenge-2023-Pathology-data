import random
import numpy as np
import pandas as pd
import skimage.io as io
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
import pickle
from tqdm import tqdm
from _hyperparameters import SEED, removing_background_config, patch_extraction_config, model_config, training_config


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def png_to_npy(process_type):
    assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

    save_dir = Path('dataset') / f'{process_type}_npy'
    save_dir.mkdir(parents=True, exist_ok=True)

    pngs_dir = Path('dataset') / 'train' if process_type == 'train' else Path('dataset') / 'test'
    pngs_dir = sorted(pngs_dir.glob('*.png'))

    pbar = tqdm(pngs_dir)
    for png_dir in pbar:
        png = png_dir.stem
        pbar.set_description(f'Now {png}.png -> {png}.npy')

        npy = io.imread(png_dir)
        np.save(save_dir / f'{png}.npy', npy)


def get_patch_metadata(process_type, save=False):
    assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

    patches_dir = Path('dataset') / f'{process_type}_patch_rs'
    patch_paths = list(patches_dir.glob('*.npy'))
    patches_name = [patch_file.stem for patch_file in patch_paths]
    patches_path = [str(patch_file) for patch_file in patch_paths]
    patches_rank = pd.Series([name.split('_')[-1] for name in patches_name]).astype(int).tolist()
    slide_name = [f"{process_type}_{name.split('_')[1]}" for name in patches_name]

    patch_metadata = pd.DataFrame({
        'Patch_name': patches_name,
        'Patch_path': patches_path,
        'Patch_rank': patches_rank,
        'Slide_name': slide_name
    })

    slide_metadata = pd.read_csv(Path(f'dataset/{process_type}_dataset.csv'))
    slides_dir = Path(f'dataset/{process_type}')
    slide_metadata['Slide_path'] = [str(slide_path) for slide_path in slides_dir.glob('*.png')]

    patch_metadata = slide_metadata.merge(patch_metadata)

    ## Risk Grouping
    patch_metadata['Risk'] = patch_metadata['Location'].apply(lambda x: 2 if x in ['heel', 'toe', 'big toe'] else (0 if x in ['finger', 'sole'] else 1))

    if process_type == 'train':
        patch_metadata.insert(len(patch_metadata.columns)-1, 'Recurrence', patch_metadata.pop('Recurrence'))

    if save:
        patch_metadata.to_csv(f'{process_type}_patch_metadata.csv', index=False)
    else:
        return patch_metadata


def get_patch_stats(risk=None, save=False):
    metadata = get_patch_metadata('train')
    if risk is not None:
        metadata = metadata[metadata['risk'] == risk]
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


def train_transforms():
    stats = get_patch_stats()

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


def test_transforms():
    stats = get_patch_stats()

    return A.Compose([
        A.Normalize(mean=stats['mean'], std=stats['std']),
        ToTensorV2()
    ])


def set_abunet_scheduler(optimizer, abunet_training_config=training_config['abunet']):
    config = abunet_training_config['scheduler']

    mode = 'min' if abunet_training_config['monitor'] == 'loss' else 'max'
    patience = abunet_training_config['early_stopping_rounds'] - 1
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=config['factor'],
        patience=patience,
        min_lr=config['min_lr'],
        verbose=config['verbose']
    )

    return scheduler


def get_all_config():
    config = {
        'remove_background_config': removing_background_config,
        'patch_extract_config': patch_extraction_config,
        'model_config': model_config,
        'training_config': training_config,
    }

    return config