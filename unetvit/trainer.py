import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
from unetvit.dataset import UVDataset
from unetvit.model import UNetViT
from utils.utils import get_patch_metadata, train_transforms, test_transforms, set_scheduler
from utils.config import training_config


class UVTrainer:
    def __init__(self, risk=None, config=training_config['unetvit']):
        self.risk = risk
        self.config = config

    def get_valid_indices(self):
        metadata = get_patch_metadata('train', self.risk)
        valid_indices = []
        n_splits = self.config['n_splits']

        stgkf = StratifiedGroupKFold(n_splits=n_splits)
        for train_idx, valid_idx in stgkf.split(metadata, metadata['Recurrence'], groups=metadata['Slide_name']):
            fold_indices = np.zeros(len(metadata), dtype=bool)
            fold_indices[valid_idx] = True
            valid_indices.append(fold_indices)

            return valid_indices

    @staticmethod
    def _train(train_loader, model, criterion, optimizer, device, epoch):
        model.to(device)
        model.train()

        loss_list, label_list, pred_list = [], [], []
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('Attention-Based MIL UNet + ViT | Training')

            patches, labels = patches.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()

            loss_list.append(loss.item())
            label_list.extend(labels.detach().cpu())
            pred_list.append(outputs.detach().cpu())

        train_loss, train_score = UVTrainer._aggregate(loss_list, label_list, pred_list, epoch, 'Training')

        return train_loss, train_score, label_list, pred_list

    @staticmethod
    def _valid(valid_loader, model, criterion, device, epoch):
        model.to(device)
        model.eval()

        loss_list, label_list, pred_list = [], [], []
        pbar = tqdm(valid_loader, total=len(valid_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('Attention-Based MIL UNet + ViT | Validation')

            patches, labels = patches.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(patches).squeeze()
                loss = criterion(outputs, labels.float())

                loss_list.append(loss.item())
                label_list.extend(labels.detach().cpu())
                pred_list.append(outputs.detach().cpu())

        valid_loss, valid_score = UVTrainer._aggregate(loss_list, label_list, pred_list, epoch, 'Validation')

        return valid_loss, valid_score, label_list, pred_list

    @staticmethod
    def _aggregate(loss_list, label_list, pred_list, epoch, mode, logging=False):
        loss = np.mean(loss_list)
        pred_final = np.where(torch.sigmoid(torch.cat(pred_list, dim=0)).numpy() > 0.5, 1, 0)
        n_pos = np.sum(pred_final == 1)

        acc = accuracy_score(y_true=label_list, y_pred=pred_final)
        score = roc_auc_score(y_true=label_list, y_score=pred_final)

        if logging:
            print(f'Epoch {epoch} {mode} | Loss: {loss:.5f}, Acc: {acc:.5f}, AUROC: {score:.5f}, #Pos: {100*n_pos/len(loss_list)}%')
        else:
            return loss, score

    @staticmethod
    def _update(epoch, current, best, monitor, model, model_save_dir, patience, risk=None):
        sota_updated = False
        if monitor == 'loss' and current < best:
            sota_updated = True
        elif monitor != 'loss' and current > best:
            sota_updated = True

        if sota_updated:
            print(f'Sota updated on epoch {epoch}, best {monitor} {best:.5f} -> {current:.5f}')
            best = current
            best_weights = deepcopy(model.state_dict())
            best_name = f'risk_{risk}_epoch_{epoch:02}.pth'
            torch.save(best_weights, str(model_save_dir / best_name))
            patience = 0
        else:
            patience += 1

        return best, patience

    def train_valid(self):
        risk = self.risk
        config = self.config
        device = config['device']
        model_save_dir = config['model_save_dir']
        model_save_dir.mkdir(parents=True, exist_ok=True)
        valid_indices = self.get_valid_indices()

        train_dataset = UVDataset(mode='train',
                                  risk=risk,
                                  valid_indices=valid_indices,
                                  transforms=train_transforms(risk))
        valid_dataset = UVDataset(mode='test',
                                  risk=risk,
                                  valid_indices=valid_indices,
                                  transforms=test_transforms(risk))
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False)

        model = UNetViT()
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        scheduler = set_scheduler(optimizer)

        best_performance = np.inf if config['monitor'] == 'loss' else 0.
        patience = 0
        for epoch in range(1, config['num_epochs']+1):
            print(f'Now on Epoch {epoch}/{config["num_epochs"]}')

            train_loss, train_score, batch_label, batch_pred = UVTrainer._train(train_loader, model, criterion, optimizer, device, epoch)
            valid_loss, valid_score, batch_label, batch_pred = UVTrainer._valid(valid_loader, model, criterion, device, epoch)
            scheduler.step(valid_loss if config['monitor'] == 'loss' else valid_score)

            UVTrainer._aggregate(train_loss, batch_label, batch_pred, epoch, 'Training', logging=True)
            UVTrainer._aggregate(valid_loss, batch_label, batch_pred, epoch, 'Validation', logging=True)

            if config['monitor'] == 'loss':
                best_performance, patience = UVTrainer._update(epoch, valid_loss, best_performance, 'loss', model, model_save_dir, patience)
            else:
                best_performance, patience = UVTrainer._update(epoch, valid_score, best_performance, 'score', model, model_save_dir, patience)

            if patience == config['early_stopping_rounds']:
                print(f'Early stopping triggered on epoch {epoch-patience}.')

                break
