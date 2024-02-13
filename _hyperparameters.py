import pickle
from pathlib import Path
import warnings
warnings.filterwarnings(action='ignore')


## Seed Setting
seed = 42


## Ready
ready_config = {
    'train_dir': Path('dataset') / 'train',
    'test_dir': Path('dataset') / 'test_public'
}

remove_background_config = {
    'split': 30,
    'threshold': 0.8
}

import numpy as np
import cv2


class BGRemover:
    def __init__(self, remove_background_config):
        self.split = remove_background_config['split']
        self.threshold = remove_background_config['threshold']

    @staticmethod
    def get_blue(slide):
        hsv_slide = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)
        hsv_slide[:, :, 0] = (hsv_slide[:, :, 0] + 10) % 180
        bgr_slide = cv2.cvtColor(hsv_slide, cv2.COLOR_HSV2BGR)
        lab_slide = cv2.cvtColor(bgr_slide, cv2.COLOR_BGR2Lab)
        _, _, b = cv2.split(lab_slide)
        return b

    def remove_background_by_direction(self, slide, direction='vertical'):
        b = BGRemover.get_blue(slide)
        if direction == 'vertical':
            axis = 0
            split = self.split
            size = slide.shape[0] // split
        elif direction == 'horizontal':
            axis = 1
            split = self.split
            size = slide.shape[1] // split

        rm_inds = []
        for i in range(0, split):
            if direction == 'vertical':
                window = b[i * size:(i + 1) * size, :]
            elif direction == 'horizontal':
                window = b[:, i * size:(i + 1) * size]

            bg_pix_ratio = np.sum((window >= 170) & (window <= 210)) / np.prod(window.shape)
            if bg_pix_ratio > self.threshold:
                rm_inds.extend(range(i * size, min((i + 1) * size, slide.shape[axis])))

        slide_bg_rm = np.delete(slide, rm_inds, axis=axis)
        return slide_bg_rm

    def remove_background(self, slide):
        slide_bg_rm_1 = self.remove_background_by_direction(slide, 'vertical')
        slide_bg_rm_2 = self.remove_background_by_direction(slide_bg_rm_1, 'horizontal')

        return slide_bg_rm_2
