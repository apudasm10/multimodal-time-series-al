import json
import os
import copy
import numpy as np
import torch
import time
import sys
from tqdm import tqdm
import random

from src.dataset import ToolTrackingDataset, ToolTrackingDataset2
from src.model import TwoStreamTCN, TCN
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from src.utils import *
from models import *
from src.acq_helpers import *
from src.acq_fn import *
from datetime import timedelta


start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {device}")

random_state = 42

random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
print("Using Random_state=", random_state)

source_folder = "./tool-tracking-data/"

dataset = ToolTrackingDataset2(
    source_path=source_folder,
    tool_name="electric_screwdriver",
    window_length=0.3,
    overlap=0.25,
    exclude_time=True
)

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



X_labeled, X_unlabeled_temp, y_labeled, y_unlabeled_temp = train_test_split(dataset_filtered, all_y_filtered, test_size=0.975, stratify=all_y)
X_unlabeled, X_test, y_unlabeled, y_test = train_test_split(X_unlabeled_temp, y_unlabeled_temp, test_size=0.2, stratify=y_unlabeled_temp)

print(f"train size: {len(y_labeled)}, val size: {len(y_unlabeled)}, test size: {len(y_test)}")

unique, counts = np.unique(y_labeled, return_counts=True)
print("[TRAIN] Unique Values:", unique)
print("[TrTRAINain] Counts:", counts)

unique, counts = np.unique(y_unlabeled, return_counts=True)
print("[VAL] Unique Values:", unique)
print("[VAL] Counts:", counts)
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
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(X_labeled, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# criterion = FocalLoss(torch.Tensor([v for v in class_counts.values()]))
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=7)

best_val_loss = float('inf')

best_model_path = os.path.join("models", "__best_model.pth")

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

exit()

# ----- Branch: Entropy -----
# entropy_model = copy.deepcopy(model).to(device)
# entropy_labeled = labeled.copy()
# entropy_labeled_loader = copy.deepcopy(labeled_loader)
# entropy_unlabeled = unlabeled.copy()
# entropy_unlabeled_loader = copy.deepcopy(unlabeled_loader)
# entropy_log_dir = os.path.join(save_root, "branch_entropy")
# entropy_scores_csv = os.path.join(entropy_log_dir, "scores.csv")


# ----- Branch: Random -----
random_model = copy.deepcopy(model).to(device)
random_labeled = labeled.copy()
random_labeled_loader = copy.deepcopy(labeled_loader)
random_unlabeled = unlabeled.copy()
random_unlabeled_loader = copy.deepcopy(unlabeled_loader)
# random_log_dir = os.path.join(save_root, "branch_random")
# random_scores_csv = os.path.join(random_log_dir, "scores.csv")

# os.makedirs(random_log_dir, exist_ok=True)
# os.makedirs(entropy_log_dir, exist_ok=True)


# ===== Random branch loop =====
for round_id in range(number_of_round):
    print(f"\n========== Random Round {round_id+1}/{number_of_round} ==========")

    unlabeled_selected, idx = select_random(random_unlabeled, B)

    random_labeled = np.append(random_labeled, unlabeled_selected)

    mask = np.ones(len(random_unlabeled), dtype=bool)
    mask[idx] = False
    random_unlabeled = random_unlabeled[mask]

    print(f"Labeled Samples: {len(random_labeled)}, Uabeled Samples: {len(random_unlabeled)} and Validation Samples: {len(val_data)}")

    # Rebuild datasets/loaders
    random_labeled_data = WoundDataset(random_labeled, "data", train_transform)
    random_unlabeled_data = WoundDataset(random_unlabeled, "data", val_transform)

    random_labeled_loader, random_unlabeled_loader, val_loader = get_loaders_label(random_labeled_data, random_unlabeled_data, val_data, batch_size)

    optimizer = torch.optim.AdamW(random_model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = Trainer(random_model, random_labeled_loader, val_loader, optimizer, scheduler=None, criterion=criterion,
                      n_classes=n_classes, device=device, save_dir=random_log_dir, scores_csv=random_scores_csv, k_best=1)

    trainer.fit(epochs_per_round, use_fdl=False)

    best_model_path = trainer.get_best_model()
    random_model.load_state_dict(torch.load(best_model_path, map_location=device))


# ===== Entropy branch loop =====
# for round_id in range(number_of_round):
#     print(f"\n========== Entropy Round {round_id+1}/{number_of_round} ==========")

#     unlabeled_selected, idx = select_acq_entropy(entropy_model, entropy_unlabeled_loader, entropy_unlabeled, B, device=device, bs=batch_size, logit_interp_to=None, image_reduction="mean", mc_passes=10)

#     entropy_labeled = np.append(entropy_labeled, unlabeled_selected)

#     mask = np.ones(len(entropy_unlabeled), dtype=bool)
#     mask[idx] = False
#     entropy_unlabeled = entropy_unlabeled[mask]

#     print(f"Labeled Samples: {len(entropy_labeled)}, Uabeled Samples: {len(entropy_unlabeled)} and Validation Samples: {len(val_data)}")

#     entropy_labeled_data = WoundDataset(entropy_labeled, "data", train_transform)
#     entropy_unlabeled_data = WoundDataset(entropy_unlabeled, "data", val_transform)

#     entropy_labeled_loader, entropy_unlabeled_loader, val_loader = get_loaders_label(entropy_labeled_data, entropy_unlabeled_data, val_data, batch_size)

#     optimizer = torch.optim.AdamW(entropy_model.parameters(), lr=lr, weight_decay=weight_decay)

#     trainer = Trainer(entropy_model, entropy_labeled_loader, val_loader, optimizer, criterion=criterion,
#                       n_classes=n_classes, device=device, save_dir=entropy_log_dir, scores_csv=entropy_scores_csv)

#     trainer.fit(epochs_per_round, use_fdl=False)

#     best_model_path = trainer.get_best_model()
#     entropy_model.load_state_dict(torch.load(best_model_path, map_location=device))

    

end = time.time()

diff = end - start
formatted = str(timedelta(seconds=int(diff)))

print("Time:", formatted)