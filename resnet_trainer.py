import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
from resnet_dataset import ResNetDataset
from resnet_model import ResNet
from _hyperparameters import training_config


class ResNetTrainer:
    def __init__(self, risk=None, resnet_training_config=training_config['resnet']):
        self.risk = risk
        self.config = resnet_training_config

    @staticmethod
    def _train(train_loader, model, criterion, optimizer, device, epoch):
        model.to(device)
        model.train()

        running_loss = 0
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('Training ResNet')

            patches, labels = patches.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch} Training Loss: {epoch_loss:.5f}')

    @staticmethod
    def _valid(valid_loader, model, criterion, device, epoch, monitor):
        model.to(device)
        model.eval()

        loss_epoch, acc_epoch, auroc_epoch, auprc_epoch = 0., 0., 0., 0.
        pbar = tqdm(valid_loader, total=len(valid_loader), leave=False)
        for patches, labels in pbar:
            pbar.set_description('Validating ResNet')

            patches = patches.to(device)

            with torch.no_grad():
                outputs = model(patches)
                y_pred = (outputs > 0.5).cpu().numpy()
                y_true = labels.float().numpy()

                loss = criterion(y_pred, labels.float())
                acc = accuracy_score(y_true=y_true, y_pred=y_pred)
                auroc = roc_auc_score(y_true=y_true, y_score=y_pred)
                auprc = average_precision_score(y_true=y_true, y_score=y_pred)

            loss_epoch += loss
            acc_epoch += acc
            auroc_epoch += auroc
            auprc_epoch += auprc

        loss_epoch = loss_epoch / len(valid_loader)
        acc_epoch = acc_epoch / len(valid_loader)
        auroc_epoch = auroc_epoch / len(valid_loader)
        auprc_epoch = auprc_epoch / len(valid_loader)
        print(f'Epoch {epoch} Validation | Loss: {loss_epoch:.5f}, Acc: {acc_epoch:.5f}, AUROC: {auroc_epoch:.5f}, AUPRC: {auprc_epoch:.5f}')

        if monitor == 'loss':
            return loss_epoch
        else:
            return auroc_epoch

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
            torch.save(best_weights, str(model_save_dir / f'epoch_{epoch:02}.pth'))
            patience = 0
        else:
            patience += 1

        return best, patience

    def train_valid(self):
        risk = self.risk
        config = self.config

        device = config['device']
        monitor = config['monitor']
        batch_size = config['batch_size']
        num_epochs = config['num_epochs']
        lr = config['lr']
        early_stopping_rounds = config['early_stopping_rounds']
        num_channels = config['num_channels'][risk]

        if risk is not None:
            model_save_dir = config['model_save_dir'] / f'risk_{risk}'
        else:
            model_save_dir = config['model_save_dir'] / 'all'
        model_save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ResNetDataset(risk=risk,
                                      process_type='train')
        valid_dataset = ResNetDataset(risk=risk,
                                      process_type='test')
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

        model = ResNet(num_channels=num_channels)
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_performance = np.inf if monitor == 'loss' else 0.
        patience = 0
        for epoch in range(1, num_epochs+1):
            print(f'Now on Epoch {epoch}/{num_epochs}')

            ResNetTrainer._train(train_loader, model, criterion, optimizer, device, epoch)
            valid_performance = ResNetTrainer._valid(valid_loader, model, criterion, device, epoch, monitor)
            best_performance, patience = ResNetTrainer._update(epoch, valid_performance, best_performance, monitor, model, model_save_dir, patience)

            if patience == early_stopping_rounds:
                print(f'Early stopping triggered on epoch {epoch - patience}.')

                break

