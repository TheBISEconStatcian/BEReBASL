import torch

from typing import Optional, Tuple, Union

from berebasl.utils.normalized_shape_tensor_ops import masked_batched_trapz
from berebasl.utils.tensor_validation import assert_tensors

_int_to_float_equiv = {
    torch.int8 : torch.float16, # float8 may be problematic for ranking which is the reason to use this
    torch.int16 : torch.float16,
    torch.int32 : torch.float32,
    torch.int64 : torch.float64
}

def batched_roc_points(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    dim: int = -1,
    ) -> Tuple[torch.Tensor, ...]:
    """
    Construct ROC curve points in a fully vectorized manner for batched inputs.

    The ROC curve is computed independently for each mini-dataset along
    dimension ``dim`` (default: last). All remaining dimensions are treated as
    batch dimensions.

    This function explicitly constructs the cumulative true positives (TPR)
    and false positives (FPR) after sorting prediction scores in descending
    order. Invalid entries (as specified by ``mask_valid`` or inferred from
    ``NaN`` values) are handled explicitly and placed at the end of the sorted
    sequence.

    **Important semantic note (invalid entries):**
    Invalid positions do not contribute to TP/FP counts but are retained in the
    output tensors. As a result, the ROC curve exhibits a **flat tail** after
    the last valid element, i.e., TPR and FPR remain constant for invalid
    positions. Downstream consumers (e.g. AUROC integration) are expected to
    ignore these regions via masking.

    Args:
        scores (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing prediction scores
            along dimension ``dim``. Must be floating point or integer.
            If ``mask_valid`` is not provided, ``NaN`` entries are treated as
            invalid.

        targets (Tensor):
            Tensor of shape ``(*batch_dims, N)`` with binary labels ``{0, 1}``
            along dimension ``dim``. Boolean tensors are automatically cast to
            ``scores.dtype``.

        mask_valid (Tensor, optional):
            Boolean tensor of shape ``(*batch_dims, N)`` indicating valid
            entries. If ``None``, validity is inferred as ``~scores.isnan()``.

        dim (int, optional):
            Dimension along which to compute ROC points. Default: ``-1``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            - **fpr**: False positive rates of shape ``(*batch_dims, N+1)``
            - **tpr**: True positive rates of shape ``(*batch_dims, N+1)``
            - **sorted_scores**: Scores sorted in descending order
            - **sorted_targets**: Targets permuted according to the sort order
            - **sorted_mask_valid**: Validity mask permuted according to the sort order

            The extra leading point corresponds to the explicit ``(0, 0)``
            start of the ROC curve.

    Raises:
        IndexError:
            If ``dim`` is not a valid dimension.

        AssertionError:
            If shapes/devices of inputs do not match.

        ValueError:
            If ``scores`` has unsupported dtype or ``mask_valid`` is not boolean.

    Notes:
        - Sorting is performed after replacing invalid scores with ``-inf`` to
          ensure they are placed at the end.
        - If a valid score is exactly ``-inf``, ordering between valid and
          invalid entries at the tail is not strictly guaranteed.
        - Division by zero (e.g. no positives or no negatives) will produce
          ``NaN`` values in TPR/FPR, matching standard AUROC conventions.

    """
    if scores.dtype == torch.bool or torch.is_complex(scores):
        raise ValueError("scores must be a floating point or integer type")
    scores_ndim = scores.dim()
    if dim not in range(-scores_ndim, scores_ndim):
        raise IndexError("dim has to be valid w. r. t. the amount of dims of scores")
    
    assert_tensors(scores, targets, tensor_names="scores, targets", 
                   checks=["same_shape", "same_device"], throw_error=True)
    
    if not torch.is_floating_point(scores):
        scores = scores.to(_int_to_float_equiv[scores.dtype])

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

    # Sort by descending score along dim
    # make sure that the non-valid are at the end (practical for score change)
    order = scores.masked_fill(~mask_valid, -float('inf')).argsort(dim=dim, descending=True)
    sorted_scores = scores.gather(dim=dim, index=order)
    sorted_targets = targets.gather(dim=dim, index=order)
    sorted_mask_valid = mask_valid.gather(dim=dim, index=order)

    

    # Count positives / negatives
    sorted_targets_for_ps = sorted_targets.masked_fill(~sorted_mask_valid, 0)
    P = sorted_targets_for_ps.sum(dim=dim, keepdim=True) # [*batch_dims, 1]
    N_valid = sorted_mask_valid.sum(dim=dim, keepdim=True)
    Q = N_valid - P                                    # [*batch_dims, 1]

    # Cumulative true / false positives along dim
    tps = torch.cumsum(sorted_targets_for_ps, dim=dim)
    fps = torch.cumsum(sorted_mask_valid.to(scores.dtype) - sorted_targets_for_ps, dim=dim)


    # Normalize to TPR / FPR
    tpr = tps / P
    fpr = fps / Q

    # Explicit (0,0) start point
    zero = tps.narrow_copy(dim, start=0, length=1).zero_() # slice tps, clone it and then make it all zeros - brilliant
    tpr = torch.cat([zero, tpr], dim=dim)
    fpr = torch.cat([zero, fpr], dim=dim)

    return fpr, tpr, sorted_scores, sorted_targets, sorted_mask_valid

def batched_auroc_from_roc_points(
        fpr: torch.Tensor, 
        tpr: torch.Tensor,
        sorted_scores: torch.Tensor,
        sorted_mask_valid: torch.Tensor,
        dim: int,
        keepdim: bool = False
) -> torch.Tensor:
    """
    Computes AU-ROC from output of ``batched_roc_points``,
    dim needs to be the same as in the call of batched_roc_points
    """
    sorted_scores = sorted_scores.movedim(dim, -1)
    sorted_mask_valid = sorted_mask_valid.movedim(dim, -1)

    valid_pair = sorted_mask_valid[..., 1:] & sorted_mask_valid[..., :-1]
    score_change = (
        (sorted_scores[..., 1:] != sorted_scores[..., :-1]) &
        valid_pair
    )
    ## Add two "true columns" one for the 0,0 point and one
    ## for the first valid score which is tautologically "a change"
    ## -> use thereby preallocated tensor instead of cat for better fusing
    mask_for_trapz =score_change.new_ones(score_change.shape[:-1] + (score_change.size(-1)+2,))
    mask_for_trapz[..., 2:] = score_change

    mask_for_trapz = mask_for_trapz.movedim(-1, dim)

    return masked_batched_trapz(tpr, fpr, mask=mask_for_trapz, dim=dim, keepdim=keepdim)
    

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

    Raises:
        IndexError:
            If ``dim`` is not a valid dimension index for ``scores``.
        AssertionError:
            If ``scores`` and ``targets`` (or ``mask_valid``) do not have
            matching shapes or devices, as checked by ``assert_tensors``.
        ValueError:
            If ``scores`` has an unsupported dtype, or if ``mask_valid`` is
            provided but is not boolean.

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
    # fpr and tpr have a higher number of dims
    # along dimension dim than sorted scores and
    # sorted
    fpr, tpr, sorted_scores, _, sorted_mask_valid = batched_roc_points(
        scores, targets, mask_valid, dim
    )

    return batched_auroc_from_roc_points(
        fpr, tpr, sorted_scores, sorted_mask_valid, dim, keepdim
    )

def batched_ks_statistic(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    dim: int = -1,
    return_thresholds: bool = False,
    keepdim: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Compute the Kolmogorov-Smirnov statistic for binary classification along
    dimension ``dim``.

    The KS statistic measures the maximum separation between the cumulative
    distribution functions of the positive and negative classes. It is defined as:

    .. math::
        \text{KS} = \max_t |F_{\text{pos}}(t) - F_{\text{neg}}(t)|

    where :math:`F_{\text{pos}}(t)` and :math:`F_{\text{neg}}(t)` are the
    empirical CDFs of scores for positive and negative samples, respectively.

    The computation is performed independently along dimension ``dim``, treating
    all other dimensions as batch dimensions.

    Args:
        scores (Tensor):
            Tensor of shape ``(*batch_dims, N)`` -asuming ``dim=-1``-
            containing predicted scores or probabilities along 
            dimension ``dim``. ``N`` denotes the dimension ``dim`` and the
            tensor can have any permutation of the shape above.
        targets (Tensor):
            Binary tensor of same shape as ``scores`` containing ground truth
            labels (0 or 1, or False/True) along dimension ``dim``.
        mask_valid (Tensor, optional):
            Boolean tensor of same shape as ``scores`` indicating which
            entries are valid. If ``None``, all non-NaN entries in ``scores``
            are considered valid. Default: ``None``.
        dim (int, optional):
            Dimension along which to compute the KS statistic. Must satisfy
            ``-scores.ndim <= dim < scores.ndim``. Default: ``-1``.
        return_thresholds (bool, optional):
            If ``True``, also return the score thresholds at which the maximum
            KS statistic occurs. Default: ``False``.
        keepdim (bool, optional):
            Whether to retain dimension ``dim`` with size 1 in the output.
            Default: ``False``.

    Returns:
        Tuple[Tensor, Optional[Tensor]]:
            Tuple of ``(ks_values, ks_thresholds)``

            If ``return_thresholds=False``: Tensor of shape ``(*batch_dims,)``
            (or ``(*batch_dims, 1)`` if ``keepdim=True``) containing the KS
            statistic for each batch element.
            
            If ``return_thresholds=True``: Tuple of two tensors:
            
            - **ks_values** (*Tensor*): KS statistics, a tensor of shape
              ``(*batch_dims,)`` if ``keepdim=False`` or ``(*batch_dims, 1)``
              (in case ``dim=-1``).
            - **ks_thresholds** (*Tensor* or ``None``): Score values at which
              the maximum separation occurs if ``return_thresholds=True``, has
              the same shape as ``ks_values``. If ``return_thresholds=False``
              it returns ``None``

    Raises:
        ValueError:
            If ``scores`` is boolean or complex, or if ``mask_valid`` is not boolean.
        IndexError:
            If ``dim`` is not a valid dimension index.
        AssertionError:
            If tensors do not have matching shapes or devices.

    Example:
        .. code-block:: python

            scores = torch.tensor([[0.1, 0.4, 0.35, 0.8],
                                   [0.2, 0.3, 0.6, 0.7]])
            targets = torch.tensor([[0, 0, 1, 1],
                                    [0, 1, 0, 1]])
            
            # Compute KS statistic along last dimension
            ks_stats, _ = batched_ks_statistic(scores, targets, dim=-1)
            
            # Also get thresholds
            ks_stats, ks_thresh = batched_ks_statistic(
                scores, targets, dim=-1, return_thresholds=True
            )
    """
    
    if scores.dtype == torch.bool or torch.is_complex(scores):
        raise ValueError("scores must be a floating point or integer type")
    
    scores_ndim = scores.dim()
    if dim not in range(-scores_ndim, scores_ndim):
        raise IndexError("dim has to be valid w. r. t. the amount of dims of scores")
    
    assert_tensors(scores, targets, tensor_names="scores, targets", 
                   checks=["same_shape", "same_device"], throw_error=True)
    
    if mask_valid is None:
        mask_valid = ~scores.isnan()
    else:
        if mask_valid.dtype != torch.bool:
            raise ValueError("mask_valid has to be bool")
        
        assert_tensors(mask_valid, scores, tensor_names="mask_valid, scores",
                       checks=["same_shape", "same_device"], throw_error=True)
    
    if targets.dtype == torch.bool:
        targets = targets.to(scores.dtype)
    
    # Normalize dim
    dim %= scores_ndim
    
    # Sort by ascending score along dim (convention: threshold t means "classify as positive if score >= t")
    order = scores.argsort(dim=dim, descending=False)
    sorted_scores = scores.gather(dim=dim, index=order)
    sorted_targets = targets.gather(dim=dim, index=order)
    sorted_mask = mask_valid.gather(dim=dim, index=order)
    
    # Apply mask: only consider valid entries
    ## avoid too any problems with  the sorted targets because of nan
    sorted_targets = sorted_targets.nan_to_num(nan=-1, posinf=-1, neginf=-1)
    sorted_targets = sorted_targets * sorted_mask
    assert ~(sorted_targets == -1).any(), "Masking still left nans or infs in targets - this breaks the ks-statistic"
    
    # Count positives and negatives (only valid entries)
    P = (sorted_targets * sorted_mask).sum(dim=dim, keepdim=True)  # Total positives
    N_total = (1 - sorted_targets) * sorted_mask
    Q = N_total.sum(dim=dim, keepdim=True)  # Total negatives
    
    # Cumulative counts along dim
    cum_pos = torch.cumsum(sorted_targets * sorted_mask, dim=dim)  # TP at each threshold
    cum_neg = torch.cumsum(N_total, dim=dim)  # FP at each threshold
    
    # CDFs: proportion of positives/negatives with score <= threshold
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    cdf_pos = cum_pos / (P + eps)
    cdf_neg = cum_neg / (Q + eps)
    
    # KS statistic: maximum absolute difference between CDFs
    ks_values = torch.abs(cdf_pos - cdf_neg)
    
    # Apply mask to KS values (invalid positions should not be considered)
    ## Nan to num NOT necessary here, as any source for that was eliminated
    ## through the eps and the nan_to_num over sorted_targets
    ks_values = ks_values.nan_to_num(nan=-1, posinf=-1) * sorted_mask
    
    # Find maximum KS value along dim
    max_ks, max_indices = ks_values.max(dim=dim, keepdim=keepdim)

    if not return_thresholds:
        return max_ks, None
    
    # Get the score threshold at which max KS occurs
    if keepdim:
        ks_thresholds = sorted_scores.gather(dim=dim, index=max_indices)
    else:
        # Need to temporarily add dim back to gather, then squeeze
        max_indices_expanded = max_indices.unsqueeze(dim)
        ks_thresholds = sorted_scores.gather(dim=dim, index=max_indices_expanded)
        ks_thresholds = ks_thresholds.squeeze(dim)
    
    return max_ks, ks_thresholds

def optimal_roc_thresholds_from_roc_points(
        fpr: torch.Tensor, 
        tpr: torch.Tensor,
        sorted_scores: torch.Tensor,
        sorted_mask_valid: torch.Tensor,
        dim: int,
        keepdim: bool = False
    ) -> torch.Tensor:
    fpr_opt, tpr_opt = 0,1 # Optimal point according according to the ROC.

    dim_size = sorted_scores.size(dim)
    fpr_relevant = fpr.narrow(dim, 1, dim_size-1)
    tpr_relevant = tpr.narrow(dim, 1, dim_size-1)
    sq_eucl_dist_to_optimum_rel = (
        fpr_relevant**2 +           # No need of substracting 0!
        (tpr_relevant-tpr_opt)**2
    )
    sq_eucl_dist_to_optimum_rel.masked_fill_(
        ~sorted_mask_valid.narrow(dim, 1, dim_size-1), 
        2 # Makes sure bigger than all valids which are in [0, \sqrt{2}]
        )
    
    idx_min_valid_eucl_dist = sq_eucl_dist_to_optimum_rel.argmin(dim, keepdim=True)

    optimal_roc_scores = sorted_scores.gather(dim, index=idx_min_valid_eucl_dist)

    if not keepdim:
        optimal_roc_scores.squeeze_(dim)

    return optimal_roc_scores
    

def optimal_roc_thresholds(
        scores: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor]=None,
        dim: int = -1,
        keepdim: bool = False
    ) -> torch.Tensor:
    fpr, tpr, sorted_scores, _, sorted_mask_valid = batched_roc_points(
        scores,
        targets,
        mask,
        dim
    )

    return optimal_roc_thresholds_from_roc_points(
        fpr, tpr, sorted_scores, sorted_mask_valid, dim, keepdim
    )

    