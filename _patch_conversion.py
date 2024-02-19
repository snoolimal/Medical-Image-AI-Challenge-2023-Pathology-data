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
    def save_rescaled_patch(process_type):
        assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        save_dir = Path('dataset') / f'{process_type}_patch_rs'
        save_dir.mkdir(parents=True, exist_ok=True)

        vanila_patches_path = Path('dataset') / f'{process_type}_patch'
        vanila_patch_paths = sorted([vanila_patch_path for vanila_patch_path in vanila_patches_path.glob('*.png')])
        pbar = tqdm(vanila_patch_paths, total=len(vanila_patch_paths))
        for vanila_patch_path in pbar:
            pbar.set_description(f'Rescaling: {vanila_patch_path.name}')
            rescaled_patch = PatchConverter._rescale_patch(vanila_patch_path)

            np.save(str(save_dir / f'{vanila_patch_path.stem}.npy'), rescaled_patch)

        pbar.close()
