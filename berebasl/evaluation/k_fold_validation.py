from math import isnan

import torch

from typing import Optional, Union

from berebasl.estimation.classifiers import Classifier

def k_fold_cv_normalized_split(
    features: torch.Tensor,  # (*batch_dims, N, F)
    labels: torch.Tensor,    # (*batch_dims, N)
    mask_valid: Optional[torch.Tensor] = None, # (*batch_dims, N), bool
    rng: Optional[torch.Generator] = None,
    k: int = 4,
    nan_lbls: Union[float, int] = float('nan'),
    nan_feats: Union[float, int] = float('nan'),
    min_bads: int = 4,
    safety_checks: bool = True
):
    """
    Batched k-fold cross-validation split for 2D data with variable numbers of
    valid observations per batch element.

    Each batch element is independently split into k folds over its valid
    observations (as determined by mask_valid). Invalid observations (e.g. padding)
    are excluded from fold assignment and replaced by nan_feats/nan_lbls sentinel
    values in the output via index-0 padding. A corresponding boolean mask is
    returned to identify valid entries in the output folds.

    The remainder when count_valids[b] is not divisible by k is distributed as
    fairly as possible: at most k-1 folds will have one fewer valid observation
    than the rest, with the short folds being the last ones along the k-axis.

    Args:
        features:      Float tensor of shape (*batch_dims, N, F) containing input
                       features. Must be a floating point tensor. batch_dims may
                       be empty (i.e. unbatched 2D input is supported).
        labels:        Tensor of shape (*batch_dims, N) containing labels.
        mask_valid:    Optional bool tensor of shape (*batch_dims, N) indicating
                       which observations are valid. If None, it is inferred from
                       labels: positions where labels == nan_lbls (or isnan if
                       nan_lbls is float('nan')) are treated as invalid.
        rng:           Optional torch.Generator for reproducible random fold
                       assignments. If None, uses the default global RNG.
        k:             Number of folds. Defaults to 4.
        nan_lbls:      Sentinel value used to fill invalid label slots in the
                       output, and to infer mask_valid when it is not provided.
                       Defaults to float('nan').
        nan_feats:     Sentinel value used to fill invalid feature slots in the
                       output. Defaults to float('nan').
        safety_checks: If True, validates that features, labels and mask_valid
                       are all torch.Tensors with consistent shapes and devices,
                       that features is floating point, and that features.dim()
                       >= 2. Defaults to True.

    Returns:
        cv_feats: Float tensor of shape ``(*batch_dims, k, fold_rows, F)``, where
                  ``fold_rows = ceil(max_N_valid / k)`` and ``max_N_valid`` is the
                  maximum count_valids across the batch. Invalid/remainder slots
                  are filled with nan_feats.
        cv_lbls:  Tensor of shape ``(*batch_dims, k, fold_rows)``. Invalid/remainder
                  slots are filled with nan_lbls.
        cv_mask:  Bool tensor of shape ``(*batch_dims, k, fold_rows)``. True where
                  the corresponding entry in cv_feats/cv_lbls is a real valid
                  observation, False for invalid or remainder-padding slots.

    Raises:
        ``AssertionError:`` If ``safety_checks`` is ``True`` and the input tensors
                            do not satisfy the shape, type, or device requirements.

    Notes:
        - The fold assignment is a random permutation of the valid indices for
          each batch element independently.
        - Remainder slots (from ceil rounding) are interleaved across folds via
          a (fold_rows, k) reshape + transpose, ensuring the imbalance is at
          most 1 valid observation between any two folds for any batch element.
        - The index-0 padding trick is used so that all invalid/remainder slots
          index into a prepended nan/False row, keeping all gather ops fully
          vectorized with no Python loops over batch dims.
        - batch_dims may be empty: unbatched inputs of shape (N, F) and (N,)
          are handled correctly throughout.
    """
    if mask_valid is None:
        mask_valid = ~(labels.isnan() if isnan(nan_lbls) else labels == nan_lbls)
    if safety_checks and not (
        (type(features) is type(labels) is type(mask_valid) is torch.Tensor) and
        torch.is_floating_point(features) and
        (features.dim() >= 2) and
        (features.dim()-1 == labels.dim() == mask_valid.dim()) and
        (features.shape[:-1] == mask_valid.shape == labels.shape) and
        (features.device==mask_valid.device==labels.device)
    ):
        raise AssertionError(
            "features, labels and mask_valid (if not None) have to be torch.Tensor"
            "with the same labels.dim() leading dimensions and be on the same device."
        )
    *batch_dims, N = labels.shape
    F = features.shape[-1]

    # --- Random ordering of valid elements first, invalids pushed to end ---
    # Add small random noise to break ties; inf for invalids pushes them last
    random_scores = torch.rand(*batch_dims, N, generator=rng, device=labels.device)
    random_scores[~mask_valid] = float('inf')  # invalids sort to end
    # argsort gives 0-based positions; +1 to reserve index 0 for nan-padding
    random_orderings = random_scores.argsort(dim=-1) + 1  # (*batch_dims, N), values in [1, N]

    # --- Pad features and labels so index 0 -> nan ---
    # features: (*batch_dims, N, F) -> (*batch_dims, N+1, F)
    nan_pad_f = features.new_full((*batch_dims, 1, F), nan_feats)
    nan_padded_feats = torch.cat([nan_pad_f, features], dim=-2)
    # labels: (*batch_dims, N) -> (*batch_dims, N+1)
    nan_pad_l = labels.new_full((*batch_dims, 1), nan_lbls)
    nan_padded_labels = torch.cat([nan_pad_l, labels], dim=-1)
    # Mask: (*batch_dims, N) -> (*batch_dims, N+1)
    false_pad_mask = mask_valid.new_zeros((*batch_dims, 1))
    false_padded_mask = torch.cat([false_pad_mask, mask_valid], dim=-1)

    # --- Per-element valid counts ---
    count_valids = mask_valid.sum(dim=-1)  # (*batch_dims,)
    max_N = count_valids.max().item()

    # Fold size: ceil(max_N / k) columns per fold
    fold_rows = (max_N + k - 1) // k  # = ceil(max_N / k)
    total_slots = k * fold_rows        # >= max_N

    # --- Pad random_orderings to total_slots with 0 (nan index) ---
    # Shape: (*batch_dims, total_slots)
    if total_slots > N:
        pad_size = total_slots - N
        zero_pad = random_orderings.new_zeros(*batch_dims, pad_size)
        padded_orderings = torch.cat([random_orderings, zero_pad], dim=-1) # Logic of unvalid at the end is kept
    else:
        padded_orderings = random_orderings[..., :total_slots]

    # Zero out any valid-but-beyond-count_valids slots
    # i.e. for each batch element, positions >= count_valids should be 0
    slot_idx = torch.arange(total_slots, device=labels.device)
    # broadcast: (*batch_dims, total_slots)
    beyond_valid = slot_idx >= count_valids.unsqueeze(-1)
    padded_orderings[beyond_valid] = 0

    # --- Reshape into (k, fold_rows) folds, interleaving remainder ---
    # reshape as (fold_rows, k) then transpose so remainder is spread across folds
    cv_idx = padded_orderings.reshape(*batch_dims, fold_rows, k).transpose(-2, -1)
    # shape: (*batch_dims, k, fold_rows)

    # --- Gather features and labels ---
    # cv_idx: (*batch_dims, k, fold_rows) -> index into dim -2 of nan_padded_feats
    # features gather: (*batch_dims, k, fold_rows, F)
    cv_idx_feats = cv_idx.unsqueeze(-1).expand(*batch_dims, k, fold_rows, F)
    cv_feats = nan_padded_feats.unsqueeze(-3).expand(*batch_dims, k, N+1, F).gather(-2, cv_idx_feats)

    # labels gather: (*batch_dims, k, fold_rows)
    cv_lbls = nan_padded_labels.unsqueeze(-2).expand(*batch_dims, k, N+1).gather(-1, cv_idx)

    # Mask gather: (*batch_dims, k, fold_rows)
    cv_mask = false_padded_mask.unsqueeze(-2).expand(*batch_dims, k, N+1).gather(-1, cv_idx)

    return cv_feats, cv_lbls, cv_mask

def train_predict_classifier_fold(
        feats_cv: torch.Tensor, 
        labels_cv: torch.Tensor, 
        mask_valid_cv: torch.Tensor, 
        classif: Classifier, 
        k: int
    ):
    if not isinstance(feats_cv, torch.Tensor) and feats_cv.dim() == 3:
        raise AssertionError(
            "K-Fold evaluation only implemented for k x 2D feats. "
            "Please do a loop over batch dimensions for higher "
            "dimensional feats and "
        )
    F = feats_cv.size(-1)
    mask_train = mask_valid_cv.clone() # [B, N_m]
    mask_train[k] &= False
    classif.reset_parameters_to_initial()
    classif.fit(
        feats_cv[mask_train].reshape(-1, F),
        labels_cv[mask_train].flatten(),
    )
    if callable(getattr(classif, "eval", None)):
        classif.eval()
    with torch.no_grad():
        mask_validation = mask_valid_cv[k] # [N]
        feats_validation = feats_cv[k]
        pred_probs_bad_k = classif.predict_proba(feats_validation[mask_validation].reshape(-1, F))[:, 1]
        normalized_probs = torch.full_like(mask_validation, fill_value=float('nan'), dtype=pred_probs_bad_k.dtype)
        normalized_probs[mask_validation] = pred_probs_bad_k
        return normalized_probs

def k_fold_evaluate_classifier_on_metric(
        feats: torch.Tensor, 
        labels: torch.Tensor,
        rng: torch.Generator,
        k_folds: int,
        min_bads: int,
        classif: Classifier,
        batched_metric : callable,
        metrics_mask_name: str = "mask_valid",
        further_metrics_kwargs: dict = {}
    ):
    if not isinstance(feats, torch.Tensor) and feats.dim() == 2:
        raise AssertionError(
            "K-Fold evaluation only implemented for 2D feats. "
            "Please do a loop over batch dimensions for higher "
            "dimensional feats and "
        )

    if metrics_mask_name in further_metrics_kwargs:
        raise RuntimeError(
            "further_metric_kwargs should not be used to set a mask"
        )
    
    feats_cv, labels_cv, mask_valid_cv = k_fold_cv_normalized_split(
        feats, 
        labels,
        rng=rng,
        k=k_folds,
        min_bad=min_bads
    )

    preds_bad = torch.stack(
        [train_predict_classifier_fold(feats_cv, labels_cv, mask_valid_cv, classif, k) 
         for k in range(k_folds)],
        dim=0
    )
    
    return batched_metric(preds_bad, labels, **{metrics_mask_name : mask_valid_cv}, **further_metrics_kwargs)
