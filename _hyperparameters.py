import numpy as np
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
        'model_name': 'resnet101',
        'pretrained': True,
        'transfer': True
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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_save_dir': Path('models/resnet'),
        'num_epochs': 30,
        'lr': 1e-3,
        'batch_size': 64,
        'monitor': 'score',
        'early_stopping_rounds': 10,
        'num_channels': {
            0: '24',
            1: '12',
            2: '24'
        },
        'aggregation': {
            0: lambda x: x.median(axis=1),
            1: lambda x: x.max(axis=1),
            2: lambda x: x.mean(axis=1)
        }
    },
    'abunet': {
        'n_splits': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_save_dir': Path('models/abunet'),
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


## Prediction
prediction_config = {
    'resnet': {
        'device': training_config['resnet']['device'],
        'batch_size': training_config['resnet']['batch_size'],
        'model_save_dir': training_config['resnet']['model_save_dir'],
        'pred_save_dir': Path('predictions/resnet')
    },
    'abunet': {
        'device': training_config['abunet']['device'],
        'batch_size': training_config['abunet']['batch_size'],
        'model_save_dir': training_config['abunet']['model_save_dir'],
        'pred_save_dir': Path('predictions/abunet')
    }
}
