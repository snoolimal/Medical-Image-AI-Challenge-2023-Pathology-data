import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import cv2
from pathlib import Path
from tqdm import tqdm
from _hyperparameters import patch_extract_config


class PatchExtractor:
    def __init__(self, patch_extract_config=patch_extract_config):
        self.patch_size = patch_extract_config['patch_size']
        self.stride = patch_extract_config['stride']
        self.topk = patch_extract_config['topk']

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

    def get_window_markers(self, slide):
        h, w, _ = slide.shape

        num_x = (w - self.patch_size) // self.stride + 1
        num_y = (h - self.patch_size) // self.stride + 1

        X = [x * self.stride for x in range(num_x)]
        Y = [y * self.stride for y in range(num_y)]

        return X, Y

    def cal_tissue_area(self, masked_patch):
        bg_area = np.sum(masked_patch == 255)
        total_area = np.prod(self.patch_size)
        tissue_area = 1 - (bg_area / total_area)

        return tissue_area

    def extract_patch(self, slide_path):
        patch_size = self.patch_size
        slide = PatchExtractor._read_slide(slide_path)
        masked_slide = PatchExtractor._mask_slide(slide)
        w_markers, h_markers = self.get_window_markers(masked_slide)

        patches = []
        scores = []
        w_pinned, h_pinned = [], []
        for h in h_markers:
            for w in w_markers:
                patch = masked_slide[h:h+patch_size, w:w+patch_size, :]
                score = self.cal_tissue_area(patch)

                patches.append(patch)
                scores.append(score)
                w_pinned.append(h)
                w_pinned.append(w)

        patches = np.array(patches)
        topk_scores = np.argsort(-np.array(scores))[:self.topk]
        topk_patches = patches[topk_scores]     # patches[topk_scores, :, :, :]
        # topk_patches_h = np.array(h_pinned)[topk_scores]
        # topk_patches_w = np.array(w_pinned)[topk_scores]

        return topk_patches

    def save_patch(self, process_type):
        assert process_type in ['train', 'test'], "Parameter 'process_type' must be either 'train' or 'test'."

        save_dir = Path('dataset') / f'{process_type}_patch'
        save_dir.mkdir(parents=True, exist_ok=True)

        slides_path = Path('dataset') / f'{process_type}_bg_rm'
        slide_paths = sorted([slide_path for slide_path in slides_path.glob('*.png')])

        pbar = tqdm(slide_paths, total=len(slide_paths))
        for slide_path in pbar:
            pbar.set_description(f'Patch Extraction: Slide {slide_path.name}')
            patches = self.extract_patch(slide_path)

            for i in range(min(self.topk, patches.shape[0])):
                top_patch = patches[i]
                cv2.imwrite(str(save_dir / f'{slide_path.stem}_{i+1}_{slide_path.suffix}'), top_patch)

        pbar.close()


PatchExtractor = PatchExtractor()
PatchExtractor.save_patch('train')
PatchExtractor.save_patch('test')
