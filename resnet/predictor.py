import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from resnet.dataset import RNDataset
from resnet.model import ResNet
from utils.utils import get_patch_metadata
from utils.config import prediction_config


class RNPredictor:
    def __init__(self, risk=None, config=prediction_config['resnet']):
        self.risk = risk
        self.config = config

    def predict(self):
        risk = self.risk
        config = self.config
        device = config['device']
        aggregation = config['aggregation'][risk]

        if risk is not None:
            model_save_dir = config['model_save_dir'] / f'risk_{risk}'
            prediction_save_dir = config['prediction_save_dir'] / f'risk_{risk}'
        else:
            model_save_dir = config['model_save_dir'] / 'all'
            prediction_save_dir = config['prediction_save_dir'] / 'all'
        best_weights = [w for w in model_save_dir.glob('*.pth')][-1]
        prediction_save_dir.mkdir(parents=True, exist_ok=True)

        test_df = get_patch_metadata('test', risk)

        test_dataset = RNDataset(metadata=test_df)
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False)

        model = ResNet(num_channels=config['num_channels'][risk])
        model.load_state_dict(best_weights)
        model.to(device)

        patch_name_list, pred_list = [], []
        for patch_names, patches in tqdm(test_loader, total=len(test_loader), desc='ResNet101 MIL | Prediction'):
            patches = patches.to(device)

            with torch.no_grad():
                outputs = model(patches)

            pred_list.extend(outputs.cpu().numpy())
            patch_name_list.extend(patch_names)

        pred_final = np.array(pred_list).reshape(-1, 5)
        pred_final = aggregation(pred_final)

        pred_df = pd.DataFrame()
        pred_df['Patch_name'] = patch_name_list
        pred_df.drop_duplicates(keep='first', ignore_index=True, inplace=True)
        pred_df['Risk'] = risk
        pred_df['Proba'] = pred_final
        pred_df.to_csv(str(prediction_save_dir)+'.csv', index=False)
