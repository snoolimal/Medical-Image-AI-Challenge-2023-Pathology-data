import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
from resnet.dataset import RNDataset
from resnet.model import ResNet
from utils.config import training_config


class RNTrainer:
    def __init__(self, risk=None, config=training_config['resnet']):
        self.risk = risk
        self.config = config

    @staticmethod
    def _train(train_loader, model, criterion, optimizer, device):
        model.to(device)
        model.train()

        running_loss = 0
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('ResNet101 MIL | Training')

            patches, labels = patches.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    @staticmethod
    def _valid(valid_loader, model, criterion, device):
        model.to(device)
        model.eval()

        running_performance = np.zeros(shape=(4,), dtype=np.float32)
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
        for i, (patches, labels) in pbar:
            pbar.set_description('ResNet101 MIL | Validation')

            patches = patches.to(device)

            with torch.no_grad():
                outputs = model(patches)
                batch_pred = (outputs > 0.5).cpu().numpy()
                batch_label = labels.float().numpy()

                running_performance = RNTrainer._get_performance(criterion, batch_pred, batch_label, running_performance)

        return running_performance

    @staticmethod
    def _get_performance(criterion, pred_list, label_list, running_performance):
        performance = {
            'loss': criterion(pred_list, label_list).item(),
            'acc': accuracy_score(y_true=pred_list, y_pred=label_list),
            'auroc': roc_auc_score(y_true=pred_list, y_score=label_list),
            'aurpc': average_precision_score(y_true=pred_list, y_score=label_list)
        }

        for i, performance in enumerate(performance.values()):
            running_performance[i] += performance

        return running_performance

    @staticmethod
    def _aggregate(running_performance, total_length, epoch, mode, logging=False):
        total_performance = running_performance / total_length
        total_loss, total_acc, total_auroc, total_auprc = total_performance

        if logging:
            print(f'Epoch {epoch} {mode} | Loss: {total_loss:.5f}, Acc: {total_acc:.5f}, AUROC: {total_auroc:.5f}, AUPRC: {total_auprc:.5f}')

        return total_loss, total_acc, total_auroc, total_auprc

    @staticmethod
    def _update(epoch, current, best, monitor, model, model_save_dir, patience):
        sota_updated = False
        if monitor == 'loss' and current < best:
            sota_updated = True
        elif monitor != 'loss' and current > best:
            sota_updated = True

        if sota_updated:
            print(f'Sota updated on epoch {epoch}, best {monitor} {best:.5f} -> {current:.5f}')
            best = current
            best_weights = deepcopy(model.state_dict())
            torch.save(best_weights, str(model_save_dir)+f'_epoch_{epoch:02}.pth')
            patience = 0
        else:
            patience += 1

        return best, patience

    def train_valid(self):
        risk = self.risk
        config = self.config
        device = config['device']
        num_channels = config['num_channels'][risk]

        if risk is not None:
            model_save_dir = config['model_save_dir'] / f'risk_{risk}'
        else:
            model_save_dir = config['model_save_dir'] / 'all'
        model_save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = RNDataset(risk=risk,
                                  mode='train')
        valid_dataset = RNDataset(risk=risk,
                                  mode='test')
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False)

        model = ResNet(num_channels=num_channels)
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        best_performance = np.inf if config['monitor'] == 'loss' else 0.
        patience = 0
        for epoch in range(1, config['num_epochs']+1):
            print(f'Now on Epoch {epoch}/{config["num_epochs"]}')

            RNTrainer._train(train_loader, model, criterion, optimizer, device)
            running_performance = RNTrainer._valid(valid_loader, model, criterion, device)
            valid_loss, _, valid_score, _ = RNTrainer._aggregate(running_performance, len(valid_loader), epoch, 'Validation', logging=False)

            if config['monitor'] == 'loss':
                best_performance, patience = RNTrainer._update(epoch, valid_loss, best_performance, 'loss', model, model_save_dir, patience)
            else:
                best_performance, patience = RNTrainer._update(epoch, valid_score, best_performance, 'score', model, model_save_dir, patience)

            if patience == config['early_stopping_rounds']:
                print(f'Early stopping triggered on epoch {epoch-patience}.')

                break
