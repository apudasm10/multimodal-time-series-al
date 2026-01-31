import random
import numpy as np
import torch
import sys
import functools
torch.load = functools.partial(torch.load, weights_only=False)
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EpochScoring, Checkpoint
from sklearn.metrics import f1_score
from src.dataset import ToolTrackingDataset2
from src.model import TCN, LSTM_CLASSIF
from activelearning.AL_cycle import plot_results, strategy_comparison
from activelearning.queries.bayesian.mc_bald import mc_bald
from activelearning.queries.bayesian.mc_max_entropy import mc_max_entropy
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.representative.diversity_query import query_diversity
from activelearning.queries.hybrid.badge import query_badge
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import time
from datetime import timedelta

start = time.time()

# --- Configuration ---

try:
    random_seed = int(sys.argv[1])
except ValueError:
    print("Please use a number for <random_state>")
    sys.exit(1)

# random_seed = 42
print(f"[INFO] Using random seed: {random_seed}")

random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- 2. Custom Collate Function ---
def tcn_collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    
    acc, gyr, mag = zip(*X_batch)
    
    acc_tensor = torch.stack(acc)
    gyr_tensor = torch.stack(gyr)
    mag_tensor = torch.stack(mag)
    
    if isinstance(y_batch[0], torch.Tensor):
        y_batch = torch.stack(y_batch).long()  # works if labels are 0-dim tensors
    else:
        y_batch = torch.tensor(y_batch, dtype=torch.long)
    
    # Return dictionary matching TCN.forward(x_acc, x_gyr, x_mag)
    return (acc_tensor, gyr_tensor, mag_tensor), y_batch


# --- Data Preparation & Filtering ---

# 1. Initialize Dataset
source_folder = "./tool-tracking-data/"
base_dataset = ToolTrackingDataset2(
    source_path=source_folder, 
    tool_name="electric_screwdriver",
    window_length=0.4,
    overlap=0.25,
    exclude_time=True
)

print(f"[INFO] Original Dataset Size: {len(base_dataset)}")

classes_to_remove = [14]
X_list = []
y_list = []

print("[INFO] Filtering data...")
for i in range(len(base_dataset)):
    X, y_tensor = base_dataset[i]
    y_val = y_tensor.item()
    
    if y_val not in classes_to_remove:
        X_list.append(X)
        y_list.append(y_val)

unique_labels = sorted(list(set(y_list)))
label_map = {v: k for k, v in enumerate(unique_labels)}

print(f"[INFO] Classes Found: {unique_labels}")
print(f"[INFO] Label Mapping: {label_map}")

# Pass 3: Convert to Numpy arrays for Skorch & Apply Mapping
# X_data will contain tuples: (acc, gyr, mag)
# y_data will contain mapped integers: 0, 1, 2...
X_data = np.empty(len(X_list), dtype=object)
y_data = np.empty(len(y_list), dtype=np.int64)

print("Number of instances by class after filtering:")
print({label: y_list.count(label) for label in unique_labels})

for i in range(len(X_list)):
    X_data[i] = X_list[i]
    y_data[i] = label_map[y_list[i]]

print(f"[INFO] Filtered Dataset Size: {len(y_data)}")

# 3. Stratified Split (Train Pool vs Test)
indices = np.arange(len(y_data))

train_idx, test_idx, y_train_labels, y_test_labels = train_test_split(
    indices, y_data, test_size=0.15, stratify=y_data, random_state=random_seed
)

X_pool = X_data[train_idx]
y_pool = y_data[train_idx]

X_test = X_data[test_idx]
y_test = y_data[test_idx]

# --- Classifier Setup ---
num_classes = len(unique_labels)
n_instances = 100  # Query batch size
print(f"[INFO] Num Classes: {num_classes}")
ROOT_OUTPUT_DIR = os.path.join("Outputs", f"LSTM{n_instances}")
os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "plots"), exist_ok=True)

checkpoint_name = os.path.join(ROOT_OUTPUT_DIR, "models", f"best_weights_seed_{random_seed}.pt")

f1_cb = EpochScoring(scoring='f1_macro', lower_is_better=False, name='valid_f1', on_train=False)
checkpoint = Checkpoint(monitor='valid_f1_best', f_params=checkpoint_name)
# model = TCN(num_classes=num_classes)

def compute_class_weights_from_labels(y, beta=0.999):
    """
    Compute effective number class weights directly from labels.

    Args:
        y: array-like, shape (num_samples,)
           Class labels (int or float, discrete)
        beta: float, smoothing factor

    Returns:
        weights: np.array, shape (num_classes,)
    """
    classes, counts = np.unique(y, return_counts=True)
    classes = classes.astype(int)

    counts = counts.astype(np.float32)
    counts = np.maximum(counts, 1.0)

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num

    weights = weights / weights.mean()

    num_classes = int(np.max(classes)) + 1
    full_weights = np.zeros(num_classes, dtype=np.float32)
    full_weights[classes] = weights

    return full_weights


class_weights = compute_class_weights_from_labels(y_pool, beta=0.999)
class_weights_tensor = torch.from_numpy(class_weights).float()

classifier = NeuralNetClassifier(
    module=LSTM_CLASSIF,
    module__input_size=3,
    module__hidden_size=32,
    module__num_layers=1,
    module__dropout_p=0.3,
    module__num_classes=num_classes,
    criterion=torch.nn.CrossEntropyLoss,
    criterion__weight=class_weights_tensor,
    optimizer=torch.optim.Adam,
    lr=2e-3,
    batch_size=16,
    max_epochs=100,
    device=device,
    
    # Custom Collate to handle the (acc, gyr, mag) tuple
    iterator_train__collate_fn=tcn_collate_fn,
    iterator_valid__collate_fn=tcn_collate_fn,
    
    # Internal validation split
    train_split=CVSplit(cv=0.15 / 0.85, stratified=True, random_state=random_seed),
    
    callbacks=[f1_cb, checkpoint],

    iterator_train__num_workers=8,   # Match your SBATCH cpus-per-task
    iterator_train__pin_memory=True, # Faster CPU -> GPU transfer
    iterator_valid__num_workers=8,
    iterator_valid__pin_memory=True,
    # verbose=1
)

# --- Initial Training (Goal Reference) ---
print("Training initial model on full pool (upper bound)...")
classifier.fit(X_pool, y_pool)
classifier.load_params(f_params=checkpoint_name)

y_pred = classifier.predict(X_test)
goal_acc = classifier.score(X_test, y_test)
goal_f1 = f1_score(y_test, y_pred, average="macro")

print(f"Goal Accuracy: {goal_acc:.4f}")
print(f"Goal F1 Score: {goal_f1:.4f}")

classifier.criterion__weight = None  # Remove class weights for AL cycles
classifier.train_split = CVSplit(cv=0.15, stratified=False, random_state=random_seed)

# --- Active Learning Setup ---
n_initial = 400
# initial_idx = np.random.choice(len(X_pool), size=n_initial, replace=False)
X_initial, X_pool_al, y_initial, y_pool_al = train_test_split(X_pool, y_pool, train_size=n_initial, stratify=y_pool, random_state=random_seed)

# X_initial = X_pool[initial_idx]
# y_initial = y_pool[initial_idx]

# # Remove initial set from the pool
# X_pool_al = np.delete(X_pool, initial_idx, axis=0)
# y_pool_al = np.delete(y_pool, initial_idx, axis=0)

print("Number of instances by class in initial set:")
print({int(k): int(v) for k, v in zip(*np.unique(y_initial, return_counts=True))})
print("Starting Active Learning Strategy Comparison...")

# Reset classifier
classifier.initialize()

scores = strategy_comparison(
    X_train=X_initial,
    y_train=y_initial,
    X_pool=X_pool_al,
    y_pool=y_pool_al,
    X_test=X_test,
    y_test=y_test,
    classifier=classifier,
    # query_type="uncertainty",
    query_strategies=[mc_max_entropy, mc_bald, query_random, query_coreset, query_diversity, query_badge],
    n_instances=[n_instances], # Query batch size
    goal_metric="f1",
    goal_metric_val=goal_f1,
    checkpoint_name=checkpoint_name,
)

image_save_path = os.path.join(ROOT_OUTPUT_DIR, "plots", f"active_learning_results_seed_{random_seed}.png")
# Plotting
print(scores)
plot_results(
    scores, 
    n_instances=[n_instances], 
    tot_samples=len(X_pool), 
    figsize=(15, 6), 
    goal_metric="f1", 
    goal_metric_val=goal_f1,
    save_path=image_save_path
)

results_df = scores[0]
results_df["goal_acc_val"] = goal_acc
results_df["goal_f1_val"] = goal_f1

filename = os.path.join(ROOT_OUTPUT_DIR, "results", f"active_learning_results_seed_{random_seed}.csv")

results_df.to_csv(filename, index=False)
print(f"[INFO] Results saved successfully to: {filename}")

end = time.time()

diff = end - start
formatted = str(timedelta(seconds=int(diff)))

print("Time:", formatted)