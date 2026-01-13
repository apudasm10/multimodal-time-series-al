import json
import os
import wandb
from src.utils import *
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, early_stopping=None, scheduler=None, save_score=False, save_dir="models", k_best=1, label_map=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.label_map = label_map

        self.save_score = save_score
        self.save_dir = save_dir
        self.k_best = k_best
        self.best_models = []
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=early_stopping)
        else:
            self.early_stopping = None

    def fit(self, epochs=1, m=0.2, n=0.1, use_fdl=False):
        self.model.to(self.device)

        for epoch in range(epochs):
            # if use_fdl:
            #     delta = linear_delta(epoch, epochs, m, n)
            # else:
            #     delta = None

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, val_macro_f1 = self.evaluate()
            print(f"Epoch {epoch+1} Results -> Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | Macro F1: {val_macro_f1:.4f}")
            if self.save_score:
                self.track_score(epoch, train_loss, train_acc, val_loss, val_acc, val_macro_f1)
            
            self.save_model(epoch, val_macro_f1)

            if self.scheduler:
                self.scheduler.step(val_loss)
            
            if self.early_stopping:
                self.early_stopping.step(val_loss)
                if self.early_stopping.should_stop:
                    print("Early stopping activated.")
                    break

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        
        for x_acc, x_gyr, x_mag, labels in loop:
            x_acc = x_acc.to(self.device).float()
            x_gyr = x_gyr.to(self.device).float()
            x_mag = x_mag.to(self.device).float()
            
            labels_mapped = [self.label_map[int(l)] for l in labels]
            labels = torch.tensor(labels_mapped, dtype=torch.long).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x_acc, x_gyr, x_mag)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
            
        avg_acc = 100 * correct / total
        avg_loss = running_loss / len(self.train_loader)
        
        return avg_loss, avg_acc
    
    def evaluate(self):
        self.model.eval()
        loss_sum = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_acc, x_gyr, x_mag, labels in self.val_loader:
                x_acc = x_acc.to(self.device).float()
                x_gyr = x_gyr.to(self.device).float()
                x_mag = x_mag.to(self.device).float()
                
                labels_mapped = [self.label_map[int(l)] for l in labels]
                labels = torch.tensor(labels_mapped, dtype=torch.long).to(self.device)
                
                outputs = self.model(x_acc, x_gyr, x_mag)
                loss = self.criterion(outputs, labels)
                
                loss_sum += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
        accuracy = 100 * correct / len(all_preds)
        
        avg_loss = loss_sum / len(self.val_loader)
        
        return avg_loss, accuracy, macro_f1

    def save_model(self, epoch, score):
        # full_dir = os.path.join("runs", self.save_dir)
        # os.makedirs(full_dir, exist_ok=True)
        model_path = os.path.join(self.save_dir, f"model{epoch+1}_f1{score:.4f}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.best_models.append((score, model_path))

        self.best_models.sort(reverse=True, key=lambda x: x[0])
        if len(self.best_models) > self.k_best:
            _, path_to_remove = self.best_models.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

    def track_score(self, epoch, train_loss, train_acc, val_loss, val_acc, val_macro_f1):
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_macro_f1
        }

        wandb.log(log_data)


    def get_best_model(self):
        return self.best_models[0][1]