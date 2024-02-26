import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from unetvit.dataset import UVDataset
from unetvit.model import UNetViT
from utils.utils import test_transforms
from utils.config import prediction_config


class UVTrainer:
    def __init__(self, risk=None, config=prediction_config['unetvit']):
        self.risk = risk
        self.config = config

    def predict(self):
        risk = self.risk
        config = self.config
        device = config['device']

        if risk is not None:
            model_save_dir = config['model_save_dir'] / f'risk_{risk}'
            prediction_save_dir = config['prediction_save_dir'] / f'risk_{risk}'
        else:
            model_save_dir = config['model_save_dir'] / 'all'
            prediction_save_dir = config['prediction_save_dir'] / 'all'
        best_weights = [w for w in model_save_dir.glob('*.pth')][-1]
        prediction_save_dir.mkdir(parents=True, exist_ok=True)

        test_dataset = UVDataset(mode='test',
                                 risk=risk,
                                 transforms=test_transforms(risk))
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False)

        model = UNetViT()
        model.load_state_dict(best_weights)
        model.to(device)

        slide_name_list, pred_list = [], []
        for slide_names, patches in tqdm(test_loader, total=len(test_loader), desc='Attention-Based MIL UNet + ViT | Prediction'):
            patches.to(device)

            with torch.no_grad():
                outputs = model(patches).squeeze()

            preds = torch.sigmoid(torch.cat(outputs.detach().cpu(), dim=0)).numpy()
            pred_list.extend(preds)
            slide_name_list.extend(slide_names)

        pred_df = pd.DataFrame({'Slide_name': slide_name_list, 'Risk': risk, 'Proba': pred_list})
        pred_df.to_csv(str(prediction_save_dir)+'.csv', index=False)
