import numpy as np
import skimage.io as io
from pathlib import Path
from tqdm import tqdm
from _hyperparameters import ready_config


class Ready:
    def __init__(self, ready_config=ready_config):
        self.train_dir = ready_config['train_dir']
        self.test_dir = ready_config['test_dir']

    def png_to_npy(self, process_type):
        assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        save_dir = Path('dataset') / f'{process_type}_npy'
        save_dir.mkdir(parents=True, exist_ok=True)

        pngs_dir = self.train_dir if process_type == 'train' else self.test_dir
        pngs_dir = sorted(pngs_dir.glob('*.png'))

        pbar = tqdm(pngs_dir)
        for png_dir in pbar:
            png = png_dir.stem
            pbar.set_description(f'Now {png}.png -> {png}.npy')

            npy = io.imread(png_dir)
            np.save(save_dir / f'{png}.npy', npy)


# Ready = Ready()
# Ready.png_to_npy('train')
# Ready.png_to_npy('test')
