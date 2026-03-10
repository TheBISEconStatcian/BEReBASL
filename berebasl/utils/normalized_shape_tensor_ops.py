from math import prod

from typing import Optional, Union

import torch

from .tensor_validation import assert_tensors

def masked_batched_trapz(
    y: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False
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


