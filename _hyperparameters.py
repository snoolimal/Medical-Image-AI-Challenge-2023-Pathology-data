import torch
from pathlib import Path
import warnings
warnings.filterwarnings(action='ignore')


## Seed Setting
SEED = 42


## Background Removing
removing_background_config = {
    'split': 30,
    'threshold': 0.8
}


## Patch Extraction
PATCH_SIZE = 512
OVERLAP_RATIO = 0.0
TOPK = 5
patch_extraction_config = {
    'patch_size': PATCH_SIZE,
    'overlap_ratio': OVERLAP_RATIO,
    'stride': int(PATCH_SIZE * (1 - OVERLAP_RATIO)),
    'tissue_area_ratio': 0.1,
    'topk': TOPK
}

## Model
model_config = {
    'resnet': {

    },
    'abunet': {
        'unet': {
            'pretrained': True,
            'transfer': True,
            'encoder_name': 'resnet101',
            'encoder_depth': 5,
            'decoder_channels': [256, 128, 64, 32, 16],
            'decoder_attention_type': 'scse'
        },
        'patch_size_to_vit': (224, 224),
        'vit': {
            'pretrained': True,
            'transfer': True,
            'model_name': 'vit_base_patch16_224'
        }
    }
}


## Training
training_config = {
    'resnet': {
        'num_epochs': 30,
        'lr': 1e-3,
        'batch_size': 64,
        'early_stopping_rounds': 10
    },
    'abunet': {
        'n_splits': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_save_dir': Path('models'),
        'num_epochs': 10,
        'batch_size': 32,
        'lr': 1e-4,
        'monitor': 'score',
        'early_stoppping_rounds': 5,
        'scheduler': {
            'factor': 0.1,
            'min_lr': 1e-6,
            'verbose': True
        }
    }
}
