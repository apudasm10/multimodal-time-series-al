import torch
from src.acq_helpers import *
import numpy as np


def select_random(unlabeled, B, random_state=42):
    np.random.seed(random_state)
    idx = np.random.choice(len(unlabeled), size=min(B, len(unlabeled)), replace=False)
    return unlabeled[idx], idx


def select_acq_entropy(model, loader, current_pool_indices, budget, device, mc_passes=2):
    """
    Selects top-k samples with highest entropy.
    
    Args:
        loader: DataLoader for the current unlabeled subset
        current_pool_indices: List/Array of actual dataset indices corresponding to the loader
    """
    scores = entropy_scores(model, loader, device=device, mc_passes=mc_passes) 
    sorted_local_idx = torch.argsort(scores, descending=True).numpy()

    actual_budget = min(budget, len(sorted_local_idx))
    top_k_local = sorted_local_idx[:actual_budget]

    selected_real_indices = [current_pool_indices[i] for i in top_k_local]
    
    return selected_real_indices