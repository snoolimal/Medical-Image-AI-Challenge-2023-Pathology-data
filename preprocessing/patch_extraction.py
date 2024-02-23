import numpy as np
import cv2
import scipy.ndimage as ndi
import skimage.io as io
from pathlib import Path
from tqdm import tqdm
from utils.config import patch_extraction_config


class PatchExtractor:
    def __init__(self, config=patch_extraction_config):
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.topk = config['topk']

    @staticmethod
    def _read_slide(slide_path):
        slide = io.imread(slide_path)

        return slide

    @staticmethod
    def _mask_slide(slide):
        lab = cv2.cvtColor(slide, cv2.COLOR_RGB2Lab)
        _, a, _ = cv2.split(lab)
        thsld = cv2.threshold(src=a,
                              thresh=127,
                              maxval=255,
                              type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        mask = np.zeros_like(a)
        mask[a < thsld] = 1
        mask[a >= thsld] = 2
        mask = ndi.binary_fill_holes(mask - 1)

        masked_slide = np.zeros_like(slide)
        masked_slide[mask == 1] = slide[np.where(mask == 1)]
        masked_slide[mask == 0] = 255

        return masked_slide

    def get_window_marker(self, slide):
        h, w, _ = slide.shape
        num_x = (w - self.patch_size) // self.stride + 1
        num_y = (h - self.patch_size) // self.stride + 1

        X = [x * self.stride for x in range(num_x)]
        Y = [y * self.stride for y in range(num_y)]

        return X, Y

    def cal_tissue_area(self, mask_patch):
        bg_area = np.sum(mask_patch == 255)
        total_area = np.prod(self.patch_size)
        tissue_area = 1 - (bg_area / total_area)

        return tissue_area

    def extract_patch(self, slide_path):
        slide = PatchExtractor._read_slide(slide_path)
        masked_slide = PatchExtractor._mask_slide(slide)
        w_markers, h_markers = self.get_window_marker(masked_slide)

        patches = []
        scores = []
        # w_pinned, h_pinned = [], []
        for h in h_markers:
            for w in w_markers:
                mask_patch = masked_slide[h:h+self.patch_size, w:w+self.patch_size, :]
                score = self.cal_tissue_area(mask_patch)

                patches.append(mask_patch)
                scores.append(score)
                # w_pinned.append(h)
                # w_pinned.append(w)

        patches = np.array(patches)
        topk_scores = np.argsort(-np.array(scores))[:self.topk]
        topk_patches = patches[topk_scores, :, :, :]
        # topk_patches_h = np.array(h_pinned)[topk_scores]
        # topk_patches_w = np.array(w_pinned)[topk_scores]

        return topk_patches

    def extract_save_patch(self, mode):
        assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

        slide_dir = Path('dataset') / f'{mode}' / f'{mode}_bg_rm'
        save_dir = slide_dir.parent / f'{mode}_patch'
        save_dir.mkdir(parents=True, exist_ok=True)

        slide_paths = sorted(slide_dir.glob('*.png'))
        pbar = tqdm(slide_paths)
        for slide_path in pbar:
            slide_name = slide_path.name
            pbar.set_description(f'Patch Extraction | {slide_name}')

            top_patches = self.extract_patch(slide_path)
            for i in range(min(top_patches.shape[0], self.topk)):
                top_patch = top_patches[i]
                patch_name = f'{slide_path.stem}_{i+1}{slide_path.suffix}'
                io.imsave(str(save_dir / patch_name), top_patch)

        pbar.close()
