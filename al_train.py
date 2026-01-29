import numpy as np
import torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EpochScoring, Checkpoint
from sklearn.metrics import f1_score
from src.dataset import ToolTrackingDataset2
from src.model import TCN
from activelearning.AL_cycle import plot_results, strategy_comparison
from activelearning.queries.bayesian.mc_bald import mc_bald
from activelearning.queries.bayesian.mc_max_entropy import mc_max_entropy
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.hybrid.badge import query_badge
from sklearn.model_selection import train_test_split


# --- Configuration ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    indices, y_data, test_size=0.2, stratify=y_data, random_state=42
)

X_pool = X_data[train_idx]
y_pool = y_data[train_idx]

X_test = X_data[test_idx]
y_test = y_data[test_idx]

# --- Classifier Setup ---
num_classes = len(unique_labels)
print(f"[INFO] Num Classes: {num_classes}")

f1_cb = EpochScoring(scoring='f1_macro', lower_is_better=False, name='valid_f1', on_train=False)
checkpoint = Checkpoint(monitor='valid_f1_best', f_params='best_tcn_weights.pt')
# model = TCN(num_classes=num_classes)

classifier = NeuralNetClassifier(
    module=TCN,
    module__num_classes=num_classes,
    criterion=torch.nn.CrossEntropyLoss,
    # criterion__weight=_,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    batch_size=16,
    max_epochs=100,
    device=device,
    
    # Custom Collate to handle the (acc, gyr, mag) tuple
    iterator_train__collate_fn=tcn_collate_fn,
    iterator_valid__collate_fn=tcn_collate_fn,
    
    # Internal validation split
    train_split=CVSplit(0.2, stratified=True, random_state=42),
    
    callbacks=[f1_cb, checkpoint],
    verbose=0
)

# --- Initial Training (Goal Reference) ---
print("Training initial model on full pool (upper bound)...")
classifier.fit(X_pool, y_pool)
classifier.load_params(f_params='best_tcn_weights.pt')

y_pred = classifier.predict(X_test)
goal_acc = classifier.score(X_test, y_test)
goal_f1 = f1_score(y_test, y_pred, average="macro")

print(f"Goal Accuracy: {goal_acc:.4f}")
print(f"Goal F1 Score: {goal_f1:.4f}")

# --- Active Learning Setup ---
n_initial = 400
# initial_idx = np.random.choice(len(X_pool), size=n_initial, replace=False)
X_initial, X_pool_al, y_initial, y_pool_al = train_test_split(X_pool, y_pool, train_size=n_initial, stratify=y_pool, random_state=42)

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
    query_strategies=[mc_max_entropy, query_random, query_coreset, query_badge],
    n_instances=[50], # Query batch size
    goal_metric="f1",
    goal_metric_val=goal_f1
)

# Plotting
print(scores)
plot_results(
    scores, 
    n_instances=[50], 
    tot_samples=len(X_pool), 
    figsize=(15, 6), 
    goal_metric="f1", 
    goal_metric_val=goal_f1,
    save_path="active_learning_results.png"
)