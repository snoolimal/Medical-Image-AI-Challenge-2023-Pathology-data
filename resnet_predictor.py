import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet_dataset import ResNetDataset
from resnet_model import ResNet
from _hyperparameters import prediction_config


class ResNetPredictor:
    def __init__(self, risk=None, resnet_prediction_config=prediction_config['resnet']):
        self.risk = risk
        self.config = resnet_prediction_config

    def predict(self):
        config = self.config
        risk = self.risk
        device = config['device']
        batch_size = config['batch_size']
        num_channels = config['num_channels'][risk]
        aggregation = config['aggregation'][risk]
        if risk is not None:
            model_save_dir = config['model_save_dir'] / f'risk_{risk}'
            pred_save_dir = config['pred_save_dir'] / f'risk_{risk}'
        else:
            model_save_dir = config['model_save_dir'] / 'all'
            pred_save_dir = config['pred_save_dir'] / 'all'
        pred_save_dir.mkdir(parents=True, exist_ok=True)
        best_weights = [w for w in model_save_dir.glob('*.pth')][-1]

        test_dataset = ResNetDataset(risk=risk,
                                     process_type='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        model = ResNet(num_channels=num_channels)
        model.load_state_dict(best_weights)
        model.to(device)

        patch_name_list, pred_list = [], []
        for patch_names, patches in tqdm(test_loader, total=len(test_loader), desc='Prediction'):
            patches = patches.to(device)

            with torch.no_grad():
                preds = model(patches)

            pred_list.extend(preds.cpu().numpy())
            patch_name_list.extend(patch_names)

        pred_agg = np.array(pred_list).reshape(-1, 5)
        pred_agg = aggregation(pred_agg)

        pred_df = pd.DataFrame()
        pred_df['Patch_name'] = patch_name_list
        pred_df.drop_duplicates(keep='first', ignore_index=True, inplace=True)
        pred_df['Risk'] = risk
        pred_df['Proba'] = pred_agg
        pred_df.to_csv(str(pred_save_dir / f'resnet_{risk}.csv'), index=False)
