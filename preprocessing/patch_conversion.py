import numpy as np
import skimage.io as io
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from pathlib import Path
from tqdm import tqdm


class PatchConverter:
    def __init__(self):
        pass

    @staticmethod
    def _rescale_patch(patch_path):
        patch = io.imread(patch_path)
        patch = patch / 255.0

        hed_patch = rgb2hed(patch)
        h = np.array(rescale_intensity(hed_patch[:, :, 0], out_range=(0, 1),
                                       in_range=(0, np.percentile(hed_patch[:, :, 0], 99))))
        d = np.array(rescale_intensity(hed_patch[:, :, 1], out_range=(0, 1),
                                       in_range=(0, np.percentile(hed_patch[:, :, 1], 99))))
        hd = np.array(rescale_intensity(hed_patch[:, :, 2], out_range=(0, 1),
                                        in_range=(0, np.percentile(hed_patch[:, :, 2], 99))))
        rescaled_patch = np.stack((h, d, hd), axis=2)

        return rescaled_patch

    @staticmethod
    def rescale_save_patch(mode):
        assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

        patch_dir = Path('dataset') / f'{mode}' / f'{mode}_patch'
        save_dir = patch_dir.parent / f'{mode}_patch_rs'
        save_dir.mkdir(parents=True, exist_ok=True)

        patch_paths = sorted(patch_dir.glob('*.png'))
        pbar = tqdm(patch_paths)
        for patch_path in pbar:
            pbar.set_description(f'Patch Conversion | {patch_path.name}')

            rescaled_patch = PatchConverter._rescale_patch(patch_path)
            np.save(str(save_dir / f'{patch_path.stem}.npy'), rescaled_patch)

        pbar.close()
