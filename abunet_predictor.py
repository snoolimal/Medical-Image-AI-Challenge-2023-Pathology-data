import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from abunet_dataset import ABUNetDataset
from abunet_model import ABUNet
from _utils import test_transforms
from _hyperparameters import prediction_config


class ABUNetPredictor:
    def __init__(self, risk, abunet_prediction_config=prediction_config['abunet']):
        self.risk = risk
        self.config = abunet_prediction_config

    def predict(self):
        config = self.config
        risk = self.risk
        device = config['device']
        model_save_dir = config['model_save_dir']
        best_weights = model_save_dir.glob('*.pth')
        pred_save_dir = config['pred_save_dir']
        pred_save_dir.mkdir(parents=True, exist_ok=True)

        test_dataset = ABUNetDataset(process_type='test',
                                     risk=risk,
                                     transforms=test_transforms(risk))
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False)

        model = ABUNet()
        model.load_state_dict(best_weights)
        model.to(device)

        pred_list = []
        patch_name_list = []
        for patch_names, patches in tqdm(test_loader, total=len(test_loader), desc='Prediction'):
            patches = patches.to(device)

            with torch.no_grad():
                preds = model(patches).squeeze()

            preds = torch.sigmoid(torch.cat(preds.detach().cpu(), dim=0)).numpy()
            pred_list.extend(preds)
            patch_name_list.extend(patch_names)

        pred_df = pd.DataFrame({'Patch_name': patch_name_list, 'Risk': risk, 'Proba': pred_list})
        pred_df.to_csv(str(pred_save_dir), f'risk_{risk}_abunet.csv', index=False)
