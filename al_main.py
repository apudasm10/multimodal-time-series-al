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
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.utils import *
from models import *
from src.acq_helpers import *
from src.acq_fn import *
from datetime import timedelta
from trainer import Trainer
import wandb


start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {device}")

with open("api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

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



X_labeled, X_unlabeled_temp, y_labeled, y_unlabeled_temp = train_test_split(dataset_filtered, all_y_filtered, test_size=0.95, stratify=all_y)
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
run = 1
exp = "initial_experiment"

train_loader = DataLoader(X_labeled, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# criterion = FocalLoss(torch.Tensor([v for v in class_counts.values()]))
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=7)

best_val_loss = float('inf')

best_model_path = os.path.join("models", "__best_model.pth")

wandb.init(
    project=f"ADLTS",
    name=f"{exp}-{run}",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE
    }
)

trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, save_dir="models", label_map=d)
trainer.fit(10)

best_model_path = trainer.get_best_model()
model.load_state_dict(torch.load(best_model_path, map_location=device))

exit()


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


end = time.time()

diff = end - start
formatted = str(timedelta(seconds=int(diff)))

print("Time:", formatted)