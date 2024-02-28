import torch
from torchvision.transforms import transforms
from pathlib import Path
import warnings
warnings.filterwarnings(action='ignore')


SEED = 42


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


model_config = {
    'resnet': {
        'model_name': 'resnet101',
        'pretrained': True,
        'transfer': True
    },
    'unetvit': {
        'unet': {
            'encoder_name': 'resnet101',
            'encoder_depth': 5,
            'decoder_channels': [256, 128, 64, 32, 16],
            'decoder_attention_type': 'scse',
            'pretrained': True,
            'transfer': True
        },
        'vit': {
            'patch_size_to_vit': (224, 224),
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,
            'transfer': True
        }
    }
}


training_config = {
    'resnet': {
        'augmentations': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        'valid_size': 0.1,
        'model_save_dir': Path('resnet') / 'weights',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 2,
        'batch_size': 64,
        'lr': 1e-3,
        'monitor': 'score',
        'early_stoppping_rounds': 5,
        'num_channels': {
            0: 24,
            1: 12,
            2: 24,
            None: 36
        }
    },
    'unetvit': {
        'n_splits': 2,
        'model_save_dir': Path('unetvit') / 'weights',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 2,
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


prediction_config = {
    'resnet': {
        'model_save_dir': training_config['resnet']['model_save_dir'],
        'prediction_save_dir': Path('resnet') / 'predictions',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': training_config['resnet']['batch_size'],
        'num_channels': training_config['resnet']['num_channels'],
        'aggregation': {
            0: lambda x: x.median(axis=1),
            1: lambda x: x.max(axis=1),
            2: lambda x: x.mean(axis=1),
            None: lambda x: x.mean(axis=1),
        }
    },
    'unetvit': {
        'model_save_dir': training_config['unetvit']['model_save_dir'],
        'prediction_save_dir': Path('unetvit') / 'predictions',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': training_config['unetvit']['batch_size'],
    }
}
