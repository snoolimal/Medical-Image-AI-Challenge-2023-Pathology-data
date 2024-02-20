import numpy as np
import pandas as pd
import random
import skimage.io as io
import torch
from pathlib import Path
from tqdm import tqdm


def seed_everything(seed=42):
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
