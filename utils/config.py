removing_background_config = {
    'split': 30,
    'threshold': 0.8
}


PATCH_SIZE = 512
OVERLAP_RATIO = 0.0
patch_extraction_config = {
    'patch_size': PATCH_SIZE,
    'overlap_ratio': OVERLAP_RATIO,
    'stride': int(PATCH_SIZE * (1 - OVERLAP_RATIO)),
    'tissue_area_ratio': 0.1,
    'topk': 5
}