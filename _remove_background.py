import numpy as np
import skimage.io as io
import cv2
from pathlib import Path
from tqdm import tqdm
from _hyperparameters import remove_background_config


class BGRemover:
    def __init__(self, remove_background_config=remove_background_config):
        self.split = remove_background_config['split']
        self.threshold = remove_background_config['threshold']

    @staticmethod
    def get_blue(slide):
        hsv_slide = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)
        hsv_slide[:, :, 0] = (hsv_slide[:, :, 0] + 10) % 180
        # bgr_slide = cv2.cvtColor(hsv_slide, cv2.COLOR_HSV2BGR)
        lab_slide = cv2.cvtColor(hsv_slide, cv2.COLOR_BGR2Lab)  # bgr_slide
        _, _, b = cv2.split(lab_slide)

        return b

    def remove_bg_indices(self, slide, direction):
        assert direction in ['vertical', 'horizontal'], "Parameter 'direction' must be either 'vertical' or 'horizontal'."
        split = self.split

        b = BGRemover.get_blue(slide)
        if direction == 'vertical':
            axis = 0
            size = slide.shape[0] // split
        elif direction == 'horizontal':
            axis = 1
            size = slide.shape[1] // split

        rm_inds = []
        for i in range(split):
            if direction == 'vertical':
                window = b[i*size:(i+1)*size, :]
            elif direction == 'horizontal':
                window = b[:, i*size:(i+1)*size]

            bg_ratio = np.sum((window >= 170) & (window <= 210)) / np.prod(window.shape)
            if bg_ratio > self.threshold:
                rm_inds.extend(range(i*size, min((i+1)*size, slide.shape[axis])))

        slide_bg_rm = np.delete(slide, rm_inds, axis=axis)

        return slide_bg_rm

    def remove_background(self, slide):
        slide = self.remove_bg_indices(slide, 'vertical')
        slide = self.remove_bg_indices(slide, 'horizontal')

        return slide

    def save_rmbg_slides(self, process_type):
        assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        save_dir = Path('dataset') / f'{process_type}_bg_rm'
        save_dir.mkdir(parents=True, exist_ok=True)

        # slides_dir = Path('dataset') / f'{process_type}_npy'
        slides_dir = Path('dataset') / f'{process_type}'
        slide_paths = sorted([slide_path for slide_path in slides_dir.glob('*.png')])
        pbar = tqdm(slide_paths)
        for slide_path in pbar:
            slide_name = slide_path.name
            pbar.set_description(f'Removing Background | {slide_name}')

            slide = cv2.imread(str(slide_path))
            slide_bg_rm = self.remove_background(slide)

            io.imsave(str(save_dir / slide_name), slide_bg_rm)


BGRemover = BGRemover()
BGRemover.save_rmbg_slides('train')
BGRemover.save_rmbg_slides('test')
