from math import prod

from typing import Optional, Union, Literal, Tuple

import torch

from .tensor_validation import assert_tensors

def insert_piecewise_breaks_along_last_dimension(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        order_by_x_first: bool = True,
        security_checks: bool = True,
    ) -> Tuple[torch.Tensor]:
    r"""
    Insert zero-height breakpoints around contiguous valid regions.

    This function expands the last dimension of ``x``, ``y`` and ``mask`` so
    that each contiguous valid region becomes an independent piecewise curve.
    Around every contiguous region of valid samples, two additional points are
    inserted:

    * an opening point immediately before the first valid sample,
    * a closing point immediately after the last valid sample.

    Both inserted points have the same ``x`` coordinate as the adjacent valid
    sample and a function value of zero. Consequently, applying the trapezoidal
    rule to the transformed tensors computes the integral over each contiguous
    region independently instead of implicitly interpolating across invalid
    regions.

    Internally, one additional leading column is allocated in the temporary
    buffers. Index ``0`` acts as a dummy sink for every scatter operation
    corresponding to an invalid position. This allows all scatter indices to
    remain non-negative without requiring conditional indexing. The dummy column
    is removed before returning.

    Args:
        x (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing the sample
            locations along the last dimension.
        y (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing the function values
            corresponding to ``x``.
        mask (Tensor):
            Boolean tensor of shape ``(*batch_dims, N)`` indicating which
            entries are valid.
        order_by_x_first (bool, optional):
            Whether to sort ``x``, ``y`` and ``mask`` by increasing ``x`` before
            inserting breakpoints. Default: ``True``.

            Keeping this enabled is generally recommended. The purpose of this
            function is to transform the input into a representation suitable
            for piecewise trapezoidal integration over the domain defined by
            ``x``. If disabled, the inserted breakpoints follow the existing
            ordering of the samples, which may produce incorrect piecewise
            integrals whenever ``x`` is not already sorted.
        security_checks (bool, optional):
            Whether to validate input shapes, devices and dtypes before
            execution. Default: ``True``.

    Returns:
        tuple[Tensor, Tensor, Tensor]:

        * **x_new** -- Tensor of shape ``(*batch_dims, M)`` containing the
          expanded sample locations.
        * **y_new** -- Tensor of shape ``(*batch_dims, M)`` containing the
          expanded function values.
        * **mask_new** -- Boolean tensor of shape ``(*batch_dims, M)``
          indicating the valid entries in the expanded representation.

        Here ``M = N + max_extra``, where ``max_extra`` denotes the maximum
        number of inserted breakpoints over all batch elements.

    Raises:
        AssertionError:
            If ``x``, ``y`` and ``mask`` have incompatible shapes or reside on
            different devices.
        ValueError:
            If ``mask`` is not boolean.

    Example:
        .. code-block:: python

            x = torch.tensor([[300., 400., 600., 900., 1000.]])
            y = torch.tensor([[0.5, 0.6, float("nan"), 0.6, 0.7]])
            mask = ~torch.isnan(y)

            x_pw, y_pw, mask_pw = insert_piecewise_breaks_along_last_dimension(
                x, y, mask
            )
    """
    if security_checks:
        assert_tensors(
            y, x, mask, tensor_names="y, x, mask",
            checks=["are_tensors", "same_device", "same_shape"]
        )
        if mask.dtype != torch.bool:
            raise ValueError("mask was expected to be boolean")
        
    *batch_dims,  N = y.shape

    if order_by_x_first:
        ordering = x.argsort(dim=-1)
        x, y, mask = [t.gather(dim=-1, index=ordering) for t in (x, y, mask)]

    # Detect the boundaries of every contiguous valid region.
    #
    # open_[..., i]  == True  iff sample i is the first valid sample of a region.
    # close[..., i] == True  iff sample i is the last valid sample of a region.
    # Last one is closed and current is open
    is_region_end = mask & torch.nn.functional.pad(~mask[..., 1:], (0,1), value=False)
    # next one is closed and current is open
    is_region_start = mask & torch.nn.functional.pad(~mask[..., :-1], (1,0), value=False)

    breakpoint_flags  = is_region_end | is_region_start

    extra_per_batch = breakpoint_flags .sum(dim=-1)
    max_extra = extra_per_batch.max().item()

    new_N = N + max_extra

    # Allocate one additional leading column.
    #
    # All invalid positions are intentionally mapped to index 0 during the
    # scatter operations below. This dummy column is discarded before
    # returning, avoiding negative indices and conditional scatter logic.
    x_new = x.new_full((*batch_dims, new_N + 1), torch.nan)
    y_new = y.new_full((*batch_dims, new_N + 1), torch.nan)
    mask_new = mask.new_zeros((*batch_dims, new_N + 1))

    # Compute the destination index of every original valid sample.
    #
    # The mapping consists of:
    #
    #   - the running count of valid samples,
    #   - one additional position for every opening breakpoint,
    #   - one additional position for every previously inserted closing
    #     breakpoint.
    #
    # Invalid samples remain mapped to index 0.
    cumsum_mask = mask.cumsum(dim=-1)

    # Compute the offset introduced by inserted breakpoints.
    #
    # Every opening breakpoint shifts the current and all subsequent valid samples
    # by one position.
    insertion_offset = is_region_start.cumsum(dim=-1)
    # Closing breakpoints shift only the samples that follow them.
    #
    # The one-column shift ensures that the closing breakpoint is inserted after
    # the current sample and before the next valid sample.
    insertion_offset[..., 1:] += is_region_end[..., :-1].cumsum(dim=-1)
    
    idxs_map_orig = (cumsum_mask + insertion_offset) * mask

    # Closing breakpoints are inserted immediately after the corresponding
    # original sample.
    #
    # Multiplication by 'close' intentionally maps every non-closing position
    # to the dummy index 0.
    idxs_map_dummies_for_next_close = (idxs_map_orig + 1) * is_region_end

    # Opening breakpoints are inserted immediately before the corresponding
    # original sample.
    #
    # Multiplication by 'open_' intentionally maps every non-opening position
    # to the dummy index 0.
    idxs_map_dummies_for_last_close = (idxs_map_orig - 1) * is_region_start


    buffers = (
        (y_new, y, True),
        (x_new, x, False),
        (mask_new, mask, False),
    )
    idx_maps = [
        (idxs_map_orig, True),
        (idxs_map_dummies_for_next_close, False),
        (idxs_map_dummies_for_last_close, False)
    ]
    
    for buffer, src, zero_dummy_for_breaks in buffers:
        for index, is_orig_map in idx_maps:
            if zero_dummy_for_breaks and not is_orig_map:
                buffer.scatter_(dim=-1, index=index, value=0)
                continue
            buffer.scatter_(dim=-1, index=index, src=src)

    return x_new[..., 1:], y_new[..., 1:], mask_new[..., 1:] # All of them are [*batch_dims, new_N]

def masked_batched_trapz(
    y: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    nans_option: Literal["interpolate", "set_to_zero_region"] = "interpolate",
    x_is_ordered: bool = False
) -> torch.Tensor:
    r"""
    Compute a masked, batched trapezoidal integration along dimension ``dim``.

    The integration is performed only over entries where ``mask`` is ``True``.
    All other entries are ignored. The operation is applied independently to
    each mini-dataset along ``dim``; all remaining dimensions are treated as
    batch dimensions.

    Internally, the integration dimension is moved to the last axis to simplify
    indexing and reshaping, and restored to its original position afterwards.

    Args:
        y (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing function values
            along dimension ``dim``.
        x (Tensor):
            Tensor of shape ``(*batch_dims, N)`` containing sample locations
            along dimension ``dim``.
        mask (Tensor):
            Boolean tensor of shape ``(*batch_dims, N)`` indicating which
            entries are valid and should contribute to the integration.
        dim (int, optional):
            Dimension along which to integrate. Must satisfy
            ``-y.ndim <= dim < y.ndim``. Default: ``-1``.
        keepdim (bool, optional):
            Whether to retain dimension ``dim`` with size 1 in the output.
            Default: ``False``.

    Returns:
        Tensor:
            Tensor of shape ``(*batch_dims,)`` if ``keepdim=False``, otherwise
            ``(*batch_dims, 1)`` with the integration result along ``dim``.

    Raises:
        IndexError:
            If ``dim`` is not a valid dimension index for the inputs.
        AssertionError:
            If ``y``, ``x`` and ``mask`` do not have matching shapes or devices.
        ValueError:
            If ``mask`` is not boolean.

    Example:
        .. code-block:: python

            y = torch.tensor([[1., 2., 3.],
                              [4., 5., 6.]])
            x = torch.tensor([[0., 1., 2.],
                              [0., 1., 2.]])
            mask = torch.tensor([[True, True, False],
                                 [True, True, True]])

            # Integrate along last dimension
            res = masked_batched_trapz(y, x, mask, dim=-1)
    """

    assert_tensors(y, x, mask, tensor_names="y, x, mask",
                   checks=["are_tensors", "same_device", "same_shape"])
    if not mask.dtype == torch.bool:
        raise ValueError("mask was expected to be boolean")
    
    inputs_ndims = y.dim()
    if dim not in range(-inputs_ndims, inputs_ndims):
        raise IndexError("dim has to be valid w. r. t. the amount of dims of inputs")
    
    y, x, mask = [ten.movedim(dim, -1) for ten in (y, x, mask)]

    if nans_option == "set_to_zero_region":
        x, y, mask = insert_piecewise_breaks_along_last_dimension(
            x, y, mask, order_by_x_first=not x_is_ordered, security_checks=False
        )
    elif nans_option != "interpolate":
        raise AssertionError(f"nans_option {nans_option} is not recognized")
    
    *batch_dims, N = y.shape

    # Flatten batch dimensions
    B = prod(batch_dims)

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

    if batch_dims:
        results_trapz = results_trapz.reshape(*batch_dims, 1).movedim(-1, dim)
        if keepdim:
            return results_trapz
        return results_trapz.squeeze(dim)
    else:
        return results_trapz if keepdim else results_trapz[0]


