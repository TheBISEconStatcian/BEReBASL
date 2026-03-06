from math import isnan

from typing import Optional, Union

import torch

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


