import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def entropy_scores_multiclass(model, unlabeled_loader, device="cuda", bs=8, logit_interp_to=None, image_reduction="mean", mc_passes=1,):
    """
    Returns:
        indices: (N,) dataset indices aligned to unlabeled set
        scores:  (N,) image-level entropy scores (higher = more uncertain)
    """
    model.eval().to(device)

    # Enable dropout during MC if requested
    if mc_passes > 1:
        def _set_drop(m):
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
        model.apply(_set_drop)

    indices_list, scores_list = [], []

    for bidx, batch in enumerate(unlabeled_loader):
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            imgs = batch[0]
            maybe_indices = batch[1] if len(batch) > 1 else None
        else:
            imgs, maybe_indices = batch, None

        imgs = imgs.to(device)

        # MC passes of softmax probabilities
        probs_accum = None
        for _ in range(mc_passes):
            out = model(imgs)
            logits = out if not isinstance(out, tuple) else out[0]  # e.g., (logits, extras)

            if logit_interp_to is not None and logits.ndim == 4:
                logits = F.interpolate(
                    logits, size=logit_interp_to, mode="bilinear", align_corners=False
                )

            pr = torch.softmax(logits, dim=1)  # (B, C, H, W)
            probs_accum = pr if probs_accum is None else (probs_accum + pr)

        probs = probs_accum / mc_passes

        # Pixelwise entropy: H = -sum p log p
        ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1)  # (B, H, W)

        # Reduce to image-level score
        if image_reduction == "mean":
            img_scores = ent.mean(dim=(1, 2))
        elif image_reduction == "max":
            img_scores = ent.amax(dim=(1, 2))
        elif image_reduction == "p95":
            img_scores = ent.flatten(1).quantile(0.95, dim=1)
        else:
            raise ValueError("image_reduction must be one of {'mean','max','p95'}")

        indices_list.append(collect_indices(bidx, imgs, maybe_indices, bs))
        scores_list.append(img_scores.detach().cpu())

    indices = torch.cat(indices_list, dim=0)
    scores  = torch.cat(scores_list, dim=0)
    return indices, scores

@torch.no_grad()
def collect_indices(batch_idx, imgs, maybe_indices, bs):
    if (maybe_indices is not None and torch.is_tensor(maybe_indices)
        and maybe_indices.ndim == 1
        and maybe_indices.dtype in (torch.int64, torch.int32)):
        return maybe_indices.detach().cpu()
    return torch.arange(batch_idx * bs, batch_idx * bs + imgs.size(0))