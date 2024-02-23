import numpy as np
import cv2
import skimage.io as io
from pathlib import Path
from tqdm import tqdm
from utils.config import removing_background_config


class BGRemover:
    def __init__(self, config=removing_background_config):
        self.split = config['split']
        self.threshold = config['threshold']

    @staticmethod
    def _get_blue(slide):
        hsv_slide = cv2.cvtColor(slide, cv2.COLOR_RGB2HSV)
        hsv_slide[:, :, 0] = (hsv_slide[:, :, 0] + 10) % 180
        bgr_slide = cv2.cvtColor(hsv_slide, cv2.COLOR_HSV2BGR)
        lab_slide = cv2.cvtColor(bgr_slide, cv2.COLOR_BGR2Lab)
        _, _, b = cv2.split(lab_slide)

        return b

    def by_remove_background(self, slide, direction):
        assert direction in ['vertical', 'horizontal'], "Parameter 'direction' must be either 'vertical' or 'horizontal."

        split = self.split
        threshold = self.threshold

        b = BGRemover._get_blue(slide)
        if direction == 'vertical':
            axis = 0
            size = slide.shape[0] // split
        elif direction == 'horizontal':
            axis = 1
            size = slide.shape[1] // split

        rm_idx_list = []
        for i in range(split):
            start = i * size
            stop = (i + 1) * size
            if direction == 'vertical':
                window = b[start:stop, :]
            elif direction == 'horizontal':
                window = b[:, start:stop]

            bg_ratio = np.sum((window >= 170) & (window <= 210)) / np.prod(window.shape)
            if bg_ratio > threshold:
                rm_idx_list.extend(range(start, min(stop, slide.shape[axis])))

        bg_rm_slide = np.delete(slide, rm_idx_list, axis=axis)

        return bg_rm_slide

    def remove_background(self, mode):
        assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

        slide_dir = Path('dataset') / f'{mode}' / f'{mode}'
        save_dir = slide_dir.parent / f'{mode}_bg_rm'
        save_dir.mkdir(parents=True, exist_ok=True)

        slide_paths = sorted(slide_dir.glob('*.png'))
        pbar = tqdm(slide_paths)
        for slide_path in pbar:
            slide_name = slide_path.name
            pbar.set_description(f'Removing Background | {slide_name}')

            slide = io.imread(str(slide_path))
            bg_rm_slide = self.by_remove_background(slide, 'vertical')
            bg_rm_slide = self.by_remove_background(bg_rm_slide, 'horizontal')

            io.imsave(str(save_dir / slide_name), bg_rm_slide)

        pbar.close()
