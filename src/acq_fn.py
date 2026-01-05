import torch
from src.acq_helpers import *


def select_random(unlabeled, B, random_state=42):
    np.random.seed(random_state)
    idx = np.random.choice(len(unlabeled), size=min(B, len(unlabeled)), replace=False)
    return unlabeled[idx], idx

def select_acq_entropy(model, unlabeled_loader, unlabeled, B, device="cuda", bs=8, logit_interp_to=None, image_reduction="mean", mc_passes=1):
    idxs, scores = entropy_scores_multiclass(model, unlabeled_loader, device=device, bs=bs, logit_interp_to=logit_interp_to, image_reduction=image_reduction, mc_passes=mc_passes)
    order = torch.argsort(scores, descending=True)
    chosen = order[:B]

    return unlabeled[chosen], idxs[chosen].tolist()