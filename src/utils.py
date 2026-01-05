import torch
import torch.nn as nn
import torch.nn.functional as F

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
    correct = 0
    total = 0
    loss_sum = 0
    
    with torch.no_grad():
        for x_acc, x_gyr, x_mag, labels in loader:
            x_acc = x_acc.to(device).float()
            x_gyr = x_gyr.to(device).float()
            x_mag = x_mag.to(device).float()
            # x_mic = x_mic.to(device).float()
            
            # --- APPLY LABEL MAPPING ---
            # Transform raw tensor labels to indices using the dict 'd'
            labels_mapped = [label_map[int(l)] for l in labels]
            labels = torch.tensor(labels_mapped, dtype=torch.long).to(device)
            
            outputs = model(x_acc, x_gyr, x_mag)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total, loss_sum / len(loader)