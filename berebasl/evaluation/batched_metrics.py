import torch

from typing import Optional

from berebasl.utils.normalized_shape_tensor_ops import masked_batched_trapz
from berebasl.utils.tensor_validation import assert_tensors

def batched_auroc(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Batched AUROC computed via explicit ROC construction with
    grouped thresholds and trapezoidal integration with optional
    masking of valid predictions.

    AUROC is computed independently along the last dimension,
    treating all leading dimensions as batch dimensions
    (analoguous to nn.Linear-style semantics).

    Matches torchmetrics.BinaryAUROC semantics.

    Args:
        scores:   Tensor of shape (*batch_dims, N), prediction scores
        targets: Tensor of shape (*batch_dims, N), binary labels {0,1}

    Returns:
        auc: Tensor of shape (*batch_dims), AUROC per mini-dataset
    """
    """
    Batched AUROC computed via explicit ROC construction with
    grouped thresholds and trapezoidal integration.

    AUROC is computed independently along the last dimension,
    treating all leading dimensions as batch dimensions
    (analoguous to nn.Linear-style semantics).

    Matches torchmetrics.BinaryAUROC semantics.

    Args:
        scores:   Tensor of shape (*batch_dims, N), prediction scores
        targets: Tensor of shape (*batch_dims, N), binary labels {0,1}

    Returns:
        auc: Tensor of shape (*batch_dims), AUROC per mini-dataset
    """
    if scores.dtype == torch.bool or torch.is_complex(scores):
        raise ValueError("scores must be a floating point or integer type")
    
    assert_tensors(scores, targets, tensor_names="scores, targets", 
                   checks=["same_shape", "same_device"], throw_error=True)
    
    
    if mask_valid is None:
        mask_valid = ~scores.isnan()
    else:
        if mask_valid.dtype!=torch.bool:
            raise ValueError("mask_valid has to be bool")
        
        assert_tensors(mask_valid, scores, tensor_names="mask_valid_scores, scores",
                       checks=["same_shape", "same_device"], throw_error=True)
    
    if targets.dtype == torch.bool:
        targets = targets.to(scores.dtype)

    *batch_dims, N = scores.shape

    # Sort by descending score
    order = scores.argsort(dim=-1, descending=True) # [*batch_dims, N]
    sorted_scores = scores.gather(dim=-1, index=order) # [*batch_dims, N]
    sorted_targets = targets.gather(dim=-1, index=order) # [*batch_dims, N]

    

    # Count positives / negatives
    P = sorted_targets.sum(dim=-1, keepdim=True) # [*batch_dims, 1]
    Q = N - P                                    # [*batch_dims, 1]

    # Cumulative true / false positives
    tps = torch.cumsum(sorted_targets, dim=-1)     # [*batch_dims, N]
    fps = torch.cumsum(1 - sorted_targets, dim=-1) # [*batch_dims, N]
    # Identify score changes (grouped thresholds)
    # +1 because of the zero to be concatenated at the beginning
    score_change = sorted_scores.new_ones((*batch_dims, N+1), dtype=bool)
    if N > 1:
        score_change[..., 2:] = sorted_scores[..., 1:] != sorted_scores[..., :-1] # [*batch_dims, N-1]
    # Normalize to TPR / FPR
    tpr = tps / P
    fpr = fps / Q

    # Explicit (0,0) start point
    zero = tps.new_zeros((*batch_dims, 1))
    tpr = torch.cat([zero, tpr], dim=-1)
    fpr = torch.cat([zero, fpr], dim=-1)

    return masked_batched_trapz(tpr, fpr, mask=score_change)