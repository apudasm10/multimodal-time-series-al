import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def entropy_scores(model, unlabeled_loader, device="cuda", mc_passes=1):
    """
    Computes entropy for the TCN model (Acc, Gyr, Mag).
    
    Args:
        mc_passes: >1 enables Monte Carlo Dropout for better uncertainty estimation.
    Returns:
        scores: (N,) tensor of entropy scores (higher = more uncertain)
    """
    model.eval().to(device)

    # --- 1. MC Dropout Setup ---
    # If using MC sampling, force Dropout layers to stay active during Eval
    if mc_passes > 1:
        def _set_drop(m):
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                m.train()
        model.apply(_set_drop)

    all_scores = []

    # --- 2. Inference Loop ---
    for batch in unlabeled_loader:
        # Unpack based on ToolTrackingDataset structure
        # Even if we don't use Mic, the loader still yields it, so we must unpack it to ignore it.
        x_acc, x_gyr, x_mag, labels = batch
        
        x_acc = x_acc.to(device).float()
        x_gyr = x_gyr.to(device).float()
        x_mag = x_mag.to(device).float()
        
        # MC Loop
        probs_accum = None
        for _ in range(mc_passes):
            # Forward pass (3 inputs only)
            logits = model(x_acc, x_gyr, x_mag) 
            
            # TCN Output shape is (Batch, Classes)
            pr = torch.softmax(logits, dim=1)
            
            probs_accum = pr if probs_accum is None else (probs_accum + pr)

        # Average probabilities across MC passes
        probs = probs_accum / mc_passes

        # --- 3. Entropy Calculation ---
        # Formula: H = -sum(p * log(p))
        # We sum across dim=1 (Classes)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) # Result shape: (Batch,)

        all_scores.append(ent.cpu())

    return torch.cat(all_scores)