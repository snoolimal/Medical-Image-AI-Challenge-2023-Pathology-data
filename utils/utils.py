import pandas as pd
from pathlib import Path


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

    if risk is not None:
        patch_metadata = patch_metadata[patch_metadata['Risk'] == risk].reset_index(drop=True)

    if mode == 'train':
        patch_metadata.insert(len(patch_metadata.columns)-1, 'Recurrence', patch_metadata.pop('Recurrence'))

    if save:
        save_dir = str(Path('dataset') / f'{mode}_patch_metadata.csv')
        patch_metadata.to_csv(save_dir, index=False)
    else:
        return patch_metadata