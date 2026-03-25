def batched_ks_statistic(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    dim: int = -1,
    return_thresholds: bool = False,
    keepdim: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            Tensor of shape ``(*batch_dims, N)`` containing predicted scores
            or probabilities along dimension ``dim``.
        targets (Tensor):
            Binary tensor of shape ``(*batch_dims, N)`` containing ground truth
            labels (0 or 1, or False/True) along dimension ``dim``.
        mask_valid (Tensor, optional):
            Boolean tensor of shape ``(*batch_dims, N)`` indicating which
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
        Tensor or Tuple[Tensor, Tensor]:
            If ``return_thresholds=False``: Tensor of shape ``(*batch_dims,)``
            (or ``(*batch_dims, 1)`` if ``keepdim=True``) containing the KS
            statistic for each batch element.
            
            If ``return_thresholds=True``: Tuple of two tensors:
            
            - **ks_values** (*Tensor*): KS statistics as described above.
            - **ks_thresholds** (*Tensor*): Score values at which the maximum
              separation occurs.

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
            ks_stats = batched_ks_statistic(scores, targets, dim=-1)
            
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
    
    N = scores.size(dim)
    
    # Sort by ascending score along dim (convention: threshold t means "classify as positive if score >= t")
    order = scores.argsort(dim=dim, descending=False)
    sorted_scores = scores.gather(dim=dim, index=order)
    sorted_targets = targets.gather(dim=dim, index=order)
    sorted_mask = mask_valid.gather(dim=dim, index=order)
    
    # Apply mask: only consider valid entries
    ## avoid too any problems with  the sorted targets because of nan
    sorted_targets = sorted_targets.nan_to_num(nan=-1, posinf=-1, neginf=-1)
    sorted_targets = sorted_targets * sorted_mask
    
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
    ks_values = ks_values * sorted_mask
    
    # Find maximum KS value along dim
    max_ks, max_indices = ks_values.max(dim=dim, keepdim=keepdim)
    
    if return_thresholds:
        # Get the score threshold at which max KS occurs
        if keepdim:
            ks_thresholds = sorted_scores.gather(dim=dim, index=max_indices)
        else:
            # Need to temporarily add dim back to gather, then squeeze
            max_indices_expanded = max_indices.unsqueeze(dim)
            ks_thresholds = sorted_scores.gather(dim=dim, index=max_indices_expanded)
            ks_thresholds = ks_thresholds.squeeze(dim)
        
        return max_ks, ks_thresholds
    
    return max_ks


def batched_ks_threshold(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:
    r"""
    Find the optimal classification threshold based on the Kolmogorov-Smirnov
    statistic along dimension ``dim``.

    This function returns the score threshold that maximizes the separation
    between the cumulative distribution functions of positive and negative
    classes. This threshold represents the point of maximum discriminatory
    power in the scorecard.

    The operation is performed independently along dimension ``dim``, treating
    all other dimensions as batch dimensions. This is particularly useful for
    cross-validation scenarios where you want to find thresholds for multiple
    folds simultaneously.

    Args:
        scores (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing predicted scores
            or probabilities along dimension ``dim``.
        targets (Tensor):
            Binary tensor of shape ``(*batch_dims, N)`` containing ground truth
            labels (0 or 1, or False/True) along dimension ``dim``.
        mask_valid (Tensor, optional):
            Boolean tensor of shape ``(*batch_dims, N)`` indicating which
            entries are valid. If ``None``, all non-NaN entries in ``scores``
            are considered valid. Default: ``None``.
        dim (int, optional):
            Dimension along which to find the optimal threshold. Must satisfy
            ``-scores.ndim <= dim < scores.ndim``. Default: ``-1``.
        keepdim (bool, optional):
            Whether to retain dimension ``dim`` with size 1 in the output.
            Default: ``False``.

    Returns:
        Tensor:
            Tensor of shape ``(*batch_dims,)`` (or ``(*batch_dims, 1)`` if
            ``keepdim=True``) containing the optimal threshold for each batch
            element. Samples with ``score >= threshold`` should be classified
            as positive (high risk / default).

    Raises:
        ValueError:
            If ``scores`` is boolean or complex, or if ``mask_valid`` is not boolean.
        IndexError:
            If ``dim`` is not a valid dimension index.
        AssertionError:
            If tensors do not have matching shapes or devices.

    Note:
        The threshold convention is: classify as positive (1) if score >= threshold.
        This follows the standard credit scoring interpretation where higher scores
        indicate higher risk of default.

    Example:
        .. code-block:: python

            # Single fold example
            scores = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.6])
            targets = torch.tensor([0, 0, 1, 1, 1])
            threshold = batched_ks_threshold(scores, targets)
            
            # Cross-validation example with k=5 folds
            # scores_cv: [5, N_val]
            # targets_cv: [5, N_val]
            thresholds_per_fold = batched_ks_threshold(
                scores_cv, targets_cv, dim=-1
            )  # Shape: [5]
            
            # Average threshold across folds
            final_threshold = thresholds_per_fold.mean()
    """
    _, thresholds = batched_ks_statistic(
        scores=scores,
        targets=targets,
        mask_valid=mask_valid,
        dim=dim,
        return_thresholds=True,
        keepdim=keepdim
    )
    return thresholds


def cv_ks_threshold(
    model: Any,
    feats_train: torch.Tensor,
    lbls_train: torch.Tensor,
    mask_train: torch.Tensor,
    feats_vali: torch.Tensor,
    lbls_vali: torch.Tensor,
    mask_vali: torch.Tensor,
    aggregate: str = "mean",
    dim: int = -1
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Compute KS-based classification thresholds across cross-validation folds.

    This function trains a model on each CV fold, computes predictions on the
    corresponding validation set, and determines the optimal threshold using
    the KS statistic. It can return either aggregated thresholds across folds
    or individual per-fold thresholds.

    Args:
        model (Any):
            A model object with ``fit(X, y, mask)`` and ``predict_proba(X)``
            methods. The model should support batched training across multiple
            folds simultaneously.
        feats_train (Tensor):
            Training features of shape ``[k, N_train, F]`` where ``k`` is the
            number of CV folds, ``N_train`` is the number of training samples
            per fold, and ``F`` is the number of features.
        lbls_train (Tensor):
            Training labels of shape ``[k, N_train]``.
        mask_train (Tensor):
            Boolean mask of shape ``[k, N_train]`` indicating valid training
            samples.
        feats_vali (Tensor):
            Validation features of shape ``[k, N_vali, F]``.
        lbls_vali (Tensor):
            Validation labels of shape ``[k, N_vali]``.
        mask_vali (Tensor):
            Boolean mask of shape ``[k, N_vali]`` indicating valid validation
            samples.
        aggregate (str, optional):
            How to aggregate thresholds across folds. Options:
            
            - ``'mean'``: Return the mean threshold across folds.
            - ``'median'``: Return the median threshold across folds.
            - ``'none'``: Return all per-fold thresholds without aggregation.
            
            Default: ``'mean'``.
        dim (int, optional):
            Dimension along which samples are arranged in validation data.
            Default: ``-1``.

    Returns:
        Tensor or Tuple[Tensor, Tensor]:
            If ``aggregate='mean'`` or ``'median'``: A scalar tensor containing
            the aggregated threshold.
            
            If ``aggregate='none'``: A tuple ``(thresholds, ks_values)`` where:
            
            - **thresholds** (*Tensor*): Shape ``[k]``, one threshold per fold.
            - **ks_values** (*Tensor*): Shape ``[k]``, KS statistic per fold.

    Example:
        .. code-block:: python

            from your_module import BatchedLogistic
            
            k_folds = 5
            bl = BatchedLogistic(...)
            
            # Get mean threshold across folds
            threshold = cv_ks_threshold(
                model=bl,
                feats_train=train_feats,
                lbls_train=train_lbls,
                mask_train=train_masks,
                feats_vali=feats_cv,
                lbls_vali=lbls_cv,
                mask_vali=mask_valid_cv,
                aggregate='mean'
            )
            
            # Get per-fold thresholds
            thresholds, ks_stats = cv_ks_threshold(
                model=bl,
                feats_train=train_feats,
                lbls_train=train_lbls,
                mask_train=train_masks,
                feats_vali=feats_cv,
                lbls_vali=lbls_cv,
                mask_vali=mask_valid_cv,
                aggregate='none'
            )
    """
    # Train model on all folds
    model.fit(feats_train, lbls_train, mask_train)
    
    # Get predictions on validation sets
    batched_probs = model.predict_proba(feats_vali)  # [k, N_vali, 2]
    
    # Extract positive class probabilities
    pos_probs = batched_probs[..., 1]  # [k, N_vali]
    
    # Compute KS thresholds for each fold
    ks_values, thresholds = batched_ks_statistic(
        scores=pos_probs,
        targets=lbls_vali,
        mask_valid=mask_vali,
        dim=dim,
        return_thresholds=True,
        keepdim=False
    )
    
    if aggregate == "mean":
        return thresholds.mean()
    elif aggregate == "median":
        return thresholds.median()
    elif aggregate == "none":
        return thresholds, ks_values
    else:
        raise ValueError(f"aggregate must be 'mean', 'median', or 'none', got '{aggregate}'")