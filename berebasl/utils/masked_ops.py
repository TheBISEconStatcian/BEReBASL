import torch

def masked_batched_trapz(y, x, mask):
    if not (isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor) and isinstance(mask, torch.Tensor)):
        raise ValueError("x, y and mask have to be tensors")
    if not mask.dtype == torch.bool:
        raise ValueError("mask was expected to be boolean")
    if not (x.device==y.device==mask.device):
        raise ValueError("x, y and mask have to be on same device")
    if not (y.shape == x.shape == mask.shape):
        raise ValueError("y, x and mask expected to have the same shape")
    
    *batch_dims, N = y.shape

    # Flatten batch dimensions
    B = int(torch.tensor(batch_dims).prod())

    y = y.reshape(B, N)
    x = x.reshape(B, N)
    mask = mask.reshape(B, N)

    gather_idx_per_batch = mask.cumsum(dim=-1) - 1 # [B, N]
    last_valid_pos_per_batch, idx_last_valid_pos_per_batch = gather_idx_per_batch.max(dim=-1) # [B]
    max_valid = last_valid_pos_per_batch.max() + 1 # =M_v
    index_shifts_per_batch = torch.arange(B, device=y.device) * max_valid # [B]
    all_flattend_idxs_padded_uniques = gather_idx_per_batch + index_shifts_per_batch.unsqueeze(-1) # [B, N]
    # Let V = mask.sum()
    flattened_idx = all_flattend_idxs_padded_uniques[mask] #[V]

    # Get the for all batches \beta the b^{\beta}_{n_{\beta}} in the flattened index space.
    idx_b_n = all_flattend_idxs_padded_uniques.gather(dim=1, index=idx_last_valid_pos_per_batch.unsqueeze(-1)).squeeze(1) #[B]
    # We only need this difference for the first batches B-1 batches, the last one has no overlapping
    diff_valids_to_max_valids = (max_valid-1) - last_valid_pos_per_batch[:-2]
    cummulated_shifts_from_filtering_per_batch = torch.nn.functional.pad(  # (0,)
        torch.cumsum(diff_valids_to_max_valids, dim=-1),
        pad=(1,0),
        mode='constant',
        value=0
    )
    idx_last_valid_per_batch_in_filtered_flatten = idx_b_n[:-1] - cummulated_shifts_from_filtering_per_batch#[:-1]

    unvalid_plusminus_idx_in_filtered_flatten = idx_last_valid_per_batch_in_filtered_flatten

    y_flat = y[mask]
    x_flat = x[mask]

    y_additions = y_flat[1:] + y_flat[:-1]
    # Set unvalid to 0, effectively avoiding any trouble when multiplying and adding
    #return unvalid_plusminus_idx_in_filtered_flatten, flattened_idx
    #print(y_additions.shape, flattened_idx.shape, unvalid_plusminus_idx_in_filtered_flatten.shape)
    y_additions[unvalid_plusminus_idx_in_filtered_flatten] = 0
    # No need to set here the invalid ones to 0 because they are being multiplied by 0 anyways
    x_substractions = x_flat[1:] - x_flat[:-1]

    # Generate container for trapz-rule
    trapz_summands = y.new_zeros((B*max_valid,)) # [B*M_v]
    # Apply trapz rule
    trapz_summands[
        flattened_idx[:-1] # last position is not defined e. g. (y_{n+1} - y_{n}) does not exist
    ] = x_substractions * y_additions / 2 # trapz_rule
    # Reshape it
    trapz_summands = trapz_summands.reshape(B, max_valid) # [B, M_v]
    # Note: trapz_summands contains at least one trailing zero at each position
    # This cost is taken to avoid having to make an index projection of flattend_idx

    trapz_results = trapz_summands.sum(dim=-1) # [B]

    return (
        trapz_results.reshape(*batch_dims) # [*batch_dims]
        if batch_dims else 
        trapz_results[0] #[1] -> singleton tensor
    )
