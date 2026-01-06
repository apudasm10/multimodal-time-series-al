import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch
import os
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of shape [num_classes]
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_term = alpha_t * focal_term

        loss = focal_term * BCE_loss
        return loss.mean() if self.reduction == 'mean' else loss
    
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True


def evaluate(model, loader, criterion, device, label_map):
    model.eval()
    loss_sum = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_acc, x_gyr, x_mag, labels in loader:
            x_acc = x_acc.to(device).float()
            x_gyr = x_gyr.to(device).float()
            x_mag = x_mag.to(device).float()
            
            labels_mapped = [label_map[int(l)] for l in labels]
            labels = torch.tensor(labels_mapped, dtype=torch.long).to(device)
            
            outputs = model(x_acc, x_gyr, x_mag)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    
    correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
    accuracy = 100 * correct / len(all_preds)
    
    avg_loss = loss_sum / len(loader)
    
    return accuracy, macro_f1, avg_loss

def track_score(epoch, train_loss, train_dice, train_iou, train_acc, val_loss, val_dice, val_iou, val_acc, save_file="training_log.txt"):
    if not os.path.exists(save_file):
        with open(save_file, 'w') as f:
            f.write("Epoch,TrainLoss,TrainDice,TrainIoU,TrainAcc,ValLoss,ValDice,ValIoU,ValAcc\n")

    with open(save_file, 'a') as f:
        f.write(f"{epoch},{train_loss:.4f},{train_dice:.4f},{train_iou:.4f},{train_acc:.4f},{val_loss:.4f},{val_dice:.4f},{val_iou:.4f},{val_acc:.4f}\n")

def linear_delta(epoch, total_epochs, delta_max=0.2, delta_min=0.01):
    t = min(max(epoch, 0), total_epochs - 1)
    frac = t/(total_epochs - 1)

    return delta_max - frac * (delta_max - delta_min)


def cosine_delta(epoch, total_epochs, delta_max=0.2, delta_min=0.01):
    return delta_min + 0.5 * (delta_max - delta_min) * (1 + np.cos(np.pi * epoch / total_epochs))