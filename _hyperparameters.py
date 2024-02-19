import warnings
warnings.filterwarnings(action='ignore')


## Seed Setting
seed = 42


## Background Removing
remove_background_config = {
    'split': 30,
    'threshold': 0.8
}


## Patch Preprocessing
PATCH_SIZE = 512
OVERLAP_RATIO = 0.0
TOPK = 5
patch_extract_config = {
    'patch_size': PATCH_SIZE,
    'overlap_ratio': OVERLAP_RATIO,
    'stride': int(PATCH_SIZE * (1 - OVERLAP_RATIO)),
    'tissue_area_ratio': 0.1,
    'topk': TOPK
}
