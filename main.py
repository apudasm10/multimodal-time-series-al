import os
import numpy as np
from src.dataset import ToolTrackingDataset, ToolTrackingDataset2
from src.model import TwoStreamTCN, TCN
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from tqdm import tqdm
from src.utils import *

source_folder = "./tool-tracking-data/"

dataset = ToolTrackingDataset2(
    source_path=source_folder,
    tool_name="electric_screwdriver",
    window_length=0.4,
    overlap=0.25,
    exclude_time=True
)

print(dataset[0][0].shape)

class_counts = {}

all_y = []

for i in range(len(dataset)):
    y = dataset[i][-1].item()
    all_y.append(y)
    v = class_counts.setdefault(y, 0)
    v += 1
    class_counts[y] = v


print("---------------------------")
print(class_counts)

d = {}

for i, j in enumerate(class_counts.keys()):
    d[j] = i

print("---------------------------")
print(d)

print("--------------")

classes_to_remove = [14]

valid_indices = [
    i for i, label in enumerate(all_y) 
    if label not in classes_to_remove
]

dataset_filtered = Subset(dataset, valid_indices)

all_y_filtered = [all_y[i] for i in valid_indices]

class_counts = {}

all_y = []

for i in range(len(dataset_filtered)):
    y = dataset_filtered[i][-1].item()
    all_y.append(y)
    v = class_counts.setdefault(y, 0)
    v += 1
    class_counts[y] = v


print("---------------------------")
print(class_counts)

d = {}

for i, j in enumerate(class_counts.keys()):
    d[j] = i

print("---------------------------")
print(d)

print("--------------")

print(f"All Classes: {np.unique(all_y, return_counts=True)}")
print(f"Remaining Classes: {np.unique(all_y_filtered, return_counts=True)}")



X_train, X_val, y_train, y_val = train_test_split(dataset_filtered, all_y_filtered, test_size=0.2, stratify=all_y)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, stratify=y_train_val)

print(f"train size: {len(y_train)}, val size: {len(y_val)}")

unique, counts = np.unique(y_train, return_counts=True)
print("[TRAIN] Unique Values:", unique)
print("[TrTRAINain] Counts:", counts)

unique, counts = np.unique(y_val, return_counts=True)
print("[VAL] Unique Values:", unique)
print("[VAL] Counts:", counts)

# unique, counts = np.unique(y_test, return_counts=True)
# print("[TEST] Unique Values:", unique)
# print("[TrTESTain] Counts:", counts)

# print(X_train[54])

# model = TwoStreamTCN(num_classes=len(unique))
model = TCN(num_classes=len(unique))

class_mapping = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    14: 7
}

print("Total Samples", len(dataset_filtered))
exit()

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(X_val, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# criterion = FocalLoss(torch.Tensor([v for v in class_counts.values()]))
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=7)



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

best_val_loss = float('inf')

best_model_path = os.path.join("models", "__best_model_three_sensors.pth")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for x_acc, x_gyr, x_mag, labels in loop:
        x_acc = x_acc.to(DEVICE).float()
        x_gyr = x_gyr.to(DEVICE).float()
        x_mag = x_mag.to(DEVICE).float()
        # x_mic = x_mic.to(DEVICE).float()

        labels_mapped = [d[int(l)] for l in labels]
        labels = torch.tensor(labels_mapped, dtype=torch.long).to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(x_acc, x_gyr, x_mag)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
    epoch_acc = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    val_acc, val_loss = evaluate(model, test_loader, criterion, DEVICE, d)
    print(f"Epoch {epoch+1} Results -> Train Acc: {epoch_acc:.2f}% | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated! Saved to {best_model_path}")
        

    early_stopping.step(val_loss)
    if early_stopping.should_stop:
        print("Early stopping activated.")
        break

print("[INFO] Training Complete.")

print("Now lets test")
test_loader_final = DataLoader(X_val, batch_size=BATCH_SIZE, shuffle=False)

model.load_state_dict(torch.load(best_model_path))

all_preds = []
all_true = []

model.eval()
with torch.no_grad():
    for x_acc, x_gyr, x_mag, labels in tqdm(test_loader_final, desc="Testing"):
        # Move inputs to device and convert to float
        x_acc = x_acc.to(DEVICE).float()
        x_gyr = x_gyr.to(DEVICE).float()
        x_mag = x_mag.to(DEVICE).float()
        # x_mic = x_mic.to(DEVICE).float()
        
        # --- APPLY LABEL MAPPING (Same as training) ---
        # We must convert raw labels (e.g., 2, 8) to indices (0, 1) using your dict 'd'
        # Note: If a label in Test wasn't in Train, this will error. 
        # (Since you stratified, this shouldn't happen).
        labels_mapped = [d[int(l)] for l in labels]
        
        # Forward pass
        outputs = model(x_acc, x_gyr, x_mag)
        
        # Get predictions (returns value, index)
        _, predicted = torch.max(outputs, 1)
        
        # Store results (move to CPU for sklearn)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(labels_mapped)

# --- METRICS ---

# 1. Overall Accuracy
test_accuracy = accuracy_score(all_true, all_preds)
print(f"\n[FINAL TEST RESULT] Accuracy: {test_accuracy * 100:.2f}%")

test_f1 = f1_score(all_true, all_preds, average='macro')
print(f"\n[FINAL TEST RESULT] Accuracy: {test_f1 * 100:.2f}%")

# 2. Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
print("\nConfusion Matrix:")
print(cm)

# 3. Classification Report (Precision, Recall, F1-Score per class)
# We reconstruct the class names from your dictionary 'd' to make the report readable
# Sort dictionary by value (index) to ensure names match the 0,1,2... order
sorted_class_names = [str(k) for k, v in sorted(d.items(), key=lambda item: item[1])]

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=sorted_class_names, zero_division=0))