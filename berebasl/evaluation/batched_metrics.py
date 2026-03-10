import torch

from typing import Optional

from berebasl.utils.normalized_shape_tensor_ops import masked_batched_trapz
from berebasl.utils.tensor_validation import assert_tensors

def batched_auroc(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:
    r"""
    Compute AUROC independently for each mini-dataset in a batch using
    explicit ROC construction with grouped thresholds and trapezoidal
    integration.

    The AUROC is computed along dimension ``dim`` (default: last). All
    remaining dimensions are treated as batch dimensions (analogous to
    ``nn.Linear`` semantics). Semantics match
    ``torchmetrics.BinaryAUROC`` for per-batch evaluation.

    Args:
        scores (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing prediction scores
            along dimension ``dim``. Must be floating point or integer.
            ``NaN`` entries are treated as invalid unless ``mask_valid`` is
            provided.
        targets (Tensor):
            Tensor of shape ``(*batch_dims, N)`` with binary labels ``{0, 1}``
            along dimension ``dim``. Boolean tensors are automatically cast to
            ``scores.dtype``.
        mask_valid (Tensor, optional):
            Boolean tensor of shape ``(*batch_dims, N)`` indicating which
            entries are valid along dimension ``dim``. If ``None``, validity
            is inferred as ``~scores.isnan()``.
        dim (int, optional):
            Dimension along which to interpret the mini-datasets and compute
            the AUROC. Default: ``-1``.

    Returns:
        Tensor:
            Tensor of shape ``(*batch_dims,)`` containing the AUROC for each
            mini-dataset along dimension ``dim``.

    Example:
        The batched AUROC matches ``torchmetrics.BinaryAUROC`` when applied
        per batch element:

        .. code-block:: python

            import torch
            from torchmetrics import BinaryAUROC

            B, N = 4, 100
            scores = torch.rand(B, N)
            targets = torch.randint(0, 2, (B, N), dtype=torch.long)
            mask_valid = torch.ones(B, N, dtype=torch.bool)

            # Our implementation
            aurocs_batched = batched_auroc(scores, targets, mask_valid, dim=-1)

            # TorchMetrics reference
            tm_auroc = BinaryAUROC(task="binary")
            aurocs_loop = [
                tm_auroc(scores[b][mask_valid[b]], targets[b][mask_valid[b]])
                for b in range(B)
            ]
            aurocs_torchmetric = torch.stack(aurocs_loop)

            # Should be (numerically) identical
            assert torch.allclose(aurocs_batched, aurocs_torchmetric)
    """

    if scores.dtype == torch.bool or torch.is_complex(scores):
        raise ValueError("scores must be a floating point or integer type")
    scores_ndim = scores.dim()
    if dim not in range(-scores_ndim, scores_ndim):
        raise AssertionError("dim has to be valid w. r. t. the amount of dims of scores")
    
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

    # Normalize dim
    dim %= scores_ndim

    N = scores.size(dim)
    score_shape_as_list = list(scores.shape)

    # Sort by descending score along dim
    order = scores.argsort(dim=dim, descending=True)
    sorted_scores = scores.gather(dim=dim, index=order)
    sorted_targets = targets.gather(dim=dim, index=order)

    

    # Count positives / negatives
    P = sorted_targets.sum(dim=dim, keepdim=True) # [*batch_dims, 1]
    Q = N - P                                    # [*batch_dims, 1]

    # Cumulative true / false positives along dim
    tps = torch.cumsum(sorted_targets, dim=dim)
    fps = torch.cumsum(1 - sorted_targets, dim=dim)

    # Identify score changes (grouped thresholds) along dim.
    # score_change has size N+1 along dim to accommodate the explicit (0, 0) point.
    sc_shape = score_shape_as_list
    sc_shape[dim] = N+1
    score_change = sorted_scores.new_ones(torch.Size(sc_shape), dtype=bool)
    if N > 1:
        sl_score_change = tuple(
            slice(None) if d_i != dim else slice(2, N+1) 
            for d_i in range(scores_ndim)
        )

        sl_next_scores = tuple(
            slice(None) if d_i != dim else slice(1, N) 
            for d_i in range(scores_ndim)
        )
        sl_prev_scores = tuple(
            slice(None) if d_i != dim else slice(0, N-1) 
            for d_i in range(scores_ndim)
        )

        score_change[sl_score_change] = sorted_scores[sl_next_scores] != sorted_scores[sl_prev_scores] # [*batch_dims, N-1]
    # Normalize to TPR / FPR
    tpr = tps / P
    fpr = fps / Q

    # Explicit (0,0) start point
    zeros_shape = score_shape_as_list
    zeros_shape[dim] = 1
    zero = tps.new_zeros(torch.Size(zeros_shape))
    tpr = torch.cat([zero, tpr], dim=dim)
    fpr = torch.cat([zero, fpr], dim=dim)

    return masked_batched_trapz(tpr, fpr, mask=score_change, dim=dim, keepdim=keepdim)