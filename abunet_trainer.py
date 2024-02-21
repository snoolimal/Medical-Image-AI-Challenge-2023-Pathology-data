import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
from abunet_dataset import ABUNetDataset
from abunet_model import ABUNet
from _utils import get_patch_metadata, train_transforms, test_transforms, set_abunet_scheduler
from _hyperparameters import training_config


class ABUNetTrainer:
    def __init__(self, risk=None, abunet_training_config=training_config['abunet']):
        self.risk = risk
        self.config = abunet_training_config

    def get_valid_indices(self, risk):
        metadata = get_patch_metadata('train', risk)
        valid_indices = []
        n_splits = self.config['n_splits']

        stgkf = StratifiedGroupKFold(n_splits=n_splits)
        for train_idx, valid_idx in stgkf.split(metadata, metadata['Recurrence'], groups=metadata['Slide_name']):
            fold_indices = np.zeros(len(metadata), dtype=bool)
            fold_indices[valid_idx] = True
            valid_indices.append(fold_indices)

        return valid_indices

    @staticmethod
    def _aggregate(loss_list, label_list, pred_list, epoch, process_type, logging=False):
        loss = np.mean(loss_list)
        final_preds = torch.sigmoid(torch.cat(pred_list, dim=0)).numpy()
        final_preds = np.where(final_preds > 0.5, 1, 0)
        n_pos = np.sum(final_preds == 1)

        acc = accuracy_score(y_true=label_list, y_pred=final_preds)
        score = roc_auc_score(y_true=label_list, y_score=final_preds)

        if logging:
            print(f'Epoch {epoch} {process_type} | Loss: {loss:.5f}, Acc: {acc:.5f}, AUROC: {score:.5f}, #Pos: {100*n_pos/len(loss_list)}%')

        return loss, score

    @staticmethod
    def _train(train_loader, model, criterion, optimizer, device, epoch):
        model.to(device)
        model.train()

        loss_list, label_list, pred_list = [], [], []
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('Training')

            patches, labels = patches.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(patches).squeeze()
            loss = criterion(preds, labels.float())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            label_list.extend(labels.detach().cpu())
            pred_list.append(preds.detach().cpu())

        train_loss, train_score = ABUNetTrainer._aggregate(loss_list, label_list, pred_list, epoch, 'Training')

        return train_loss, train_score, label_list, pred_list

    @staticmethod
    def _valid(valid_loader, model, criterion, device, epoch):
        model.to(device)
        model.eval()

        loss_list, label_list, pred_list = [], [], []
        pbar = tqdm(valid_loader, total=len(valid_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('Validation')

            patches, labels = patches.to(device), labels.to(device)

            with torch.no_grad():
                preds = model(patches).squeeze()
                loss = criterion(preds, labels.float())

                loss_list.append(loss.item())
                label_list.extend(labels.detach().cpu())
                pred_list.append(preds.detach().cpu())

        valid_loss, valid_score = ABUNetTrainer._aggregate(loss_list, label_list, pred_list, epoch, 'Validation')

        return valid_loss, valid_score, label_list, pred_list

    def update(self, epoch, current, best, monitor, model, model_save_dir, patience):
        risk = self.risk

        sota_updated = False
        if monitor == 'loss' and current < best:
            sota_updated = True
        elif monitor != 'loss' and current > best:
            sota_updated = True

        if sota_updated:
            print(f'Sota updated on epoch {epoch}, best {monitor} {best:.5f} -> {current:.5f}')
            best = current
            best_weights = deepcopy(model.state_dict())
            torch.save(best_weights, str(model_save_dir / f'epoch_{epoch:02}.pth'))
            patience = 0
        else:
            patience += 1

        return best, patience

    def train_valid(self):
        risk = self.risk
        valid_indices = self.get_valid_indices(risk)

        config = self.config
        device = config['device']
        num_epochs = config['num_epochs']
        monitor = config['monitor']
        if risk is not None:
            model_save_dir = config['model_save_dir'] / f'risk_{risk}'
        else:
            model_save_dir = config['model_save_dir'] / 'all'
        model_save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ABUNetDataset(process_type='train',
                                      risk=risk,
                                      valid_indices=valid_indices,
                                      transforms=train_transforms(risk))
        valid_dataset = ABUNetDataset(process_type='test',
                                      risk=risk,
                                      valid_indices=valid_indices,
                                      transforms=test_transforms(risk))
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False)

        model = ABUNet()
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        scheduler = set_abunet_scheduler(optimizer)

        best_performance = np.inf if monitor == 'loss' else 0.
        patience = 0
        for epoch in range(1, num_epochs+1):
            print(f'Now on Epoch {epoch}/{num_epochs}')

            train_loss, train_score, batch_label, batch_pred = ABUNetTrainer._train(train_loader, model, criterion, optimizer, device, epoch)
            valid_loss, valid_score, batch_label, batch_pred = ABUNetTrainer._valid(valid_loader, model, criterion, device, epoch)
            scheduler.step(valid_loss if monitor == 'loss' else valid_score)

            ABUNetTrainer._aggregate(train_loss, train_score, batch_label, epoch, 'Training', logging=True)
            ABUNetTrainer._aggregate(valid_loss, valid_score, batch_label, epoch, 'Validation', logging=True)

            if monitor == 'loss':
                best_performance, patience = self.update(epoch, valid_loss, best_performance, 'loss', model, model_save_dir, patience)
            else:
                best_performance, patience = self.update(epoch, valid_score, best_performance, 'score', model, model_save_dir, patience)

            if patience == config['early_stopping_rounds']:
                print(f'Early stopping triggered on epoch {epoch-patience}.')

                break


