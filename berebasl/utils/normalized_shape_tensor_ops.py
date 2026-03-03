import torch

from typing import Union

from .tensor_validation import assert_tensors

def masked_batched_trapz(y, x, mask):
    assert_tensors(y, x, mask, tensor_names="y, x, mask",
                   checks=["are_tensors", "same_device", "same_shape"])
    if not mask.dtype == torch.bool:
        raise ValueError("mask was expected to be boolean")
    
    *batch_dims, N = y.shape

    # Flatten batch dimensions
    B = int(torch.tensor(batch_dims).prod())

    y = y.reshape(B, N)
    x = x.reshape(B, N)
    mask = mask.reshape(B, N)

    valid_count_per_batch = mask.sum(dim=-1)
    cumsums_valid_counts_batch = valid_count_per_batch.cumsum(dim=-1) - 1
    unvalid_summands_idx = cumsums_valid_counts_batch[:-1]

    y_flat = y[mask]
    x_flat = x[mask]

    y_additions = y_flat[1:] + y_flat[:-1]
    # No need to set here the invalid ones to 0 because they are being multiplied by 0 anyways
    x_substractions = x_flat[1:] - x_flat[:-1]

    flat_summands_trapz = x_substractions * y_additions / 2
    # Set unvalid to 0, effectively avoiding any mistakes when adding
    flat_summands_trapz[unvalid_summands_idx] = 0

    batch_idx_summands = torch.arange(B, device=y.device).unsqueeze(-1).expand(B, N)[mask][:-1]

    results_trapz = flat_summands_trapz.new_zeros((B,))
    results_trapz.scatter_add_(dim=0, index=batch_idx_summands, src=flat_summands_trapz)


    return (
        results_trapz.reshape(*batch_dims) # [*batch_dims]
        if batch_dims else 
        results_trapz[0] #[1] -> singleton tensor
    )

def k_fold_cv_normalized_split(features: torch.Tensor, labels: torch.Tensor, 
                               rng: torch.Generator, k: int = 4, min_bad: int = 4,
                               nan_lbls : Union[float, int] = float('nan'),
                               safety_checks: bool = True):
    if safety_checks and (not 
            (features.dim()-1 == labels.dim() == 1) and 
            (features.size(0)==labels.size(0)) and
            (features.device==labels.device)):
        raise AssertionError("Features expected to be 2d and labels 1d and have same leading dimension")
    
    N = labels.size(0)
    N_perm = torch.randperm(N, generator=rng, device=features.device) + 1
    
    nan_padded_feats = torch.nn.functional.pad(features, pad=(0,0,1,0))
    nan_padded_labels = torch.nn.functional.pad(labels, pad=(1,0), value=nan_lbls)

    
    N_mod_k = N % k
    cv_idx = N_perm.reshape(k, N//k) if N_mod_k == 0 else torch.cat([
        N_perm[:-N_mod_k].reshape(k, N//k),
        torch.nn.functional.pad(N_perm[-N_mod_k:], pad=(k-N_mod_k, 0), value=0).unsqueeze(-1)
    ], dim=-1)

    cv_feats = nan_padded_feats[cv_idx]
    cv_lbls = nan_padded_labels[cv_idx]

    return cv_feats, cv_lbls