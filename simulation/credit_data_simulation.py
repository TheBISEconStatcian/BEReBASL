import torch

from typing import Optional, Tuple

import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
proj_root_path = os.path.abspath(os.path.join(current_file_dir, ".."))
if proj_root_path not in sys.path:
    sys.path.append(proj_root_path)

from simulation.gaussian_mixture import (
    eigen_decomp_proj_to_pd,
    GaussianMixture,
    random_vcov_matrix
)

def generate_sigma_bad_and_good(
    k: int,
    proportion_var_dif: float,
    generator: torch.Generator,
    var_range: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a pair of covariance matrices: one 'good' baseline and one 'bad' perturbed version.

    The construction proceeds as follows:

    1. Generate two baseline covariance matrices using ``random_vcov_matrix``.
    2. Sample a random mask over the upper-triangular entries (including diagonal).
    3. Copy selected entries from the 'good' matrix into the 'bad' matrix, leaving
       others perturbed.
    4. Reflect the upper-triangular entries to the lower-triangular part to ensure symmetry.
    5. Project the 'bad' matrix onto the positive definite cone using
       :func:`eigen_decomp_proj_to_pd`.

    Args:
        k (int): Dimension of the covariance matrices.
        proportion_var_dif (float): Probability of keeping an entry different between
            the 'bad' and 'good' matrices.
        generator (torch.Generator): Random number generator for reproducibility.
        var_range (Tuple[float, float], optional): Range for diagonal variances.
            Defaults to (0.0, 1.0).
        eps (float, optional): Small diagonal perturbation to ensure positive definiteness.
            Defaults to ``1e-6``.
        device (torch.device, optional): Device for tensor allocation. Defaults to CPU.
        dtype (torch.dtype, optional): Data type of the returned tensors. Defaults to ``torch.float64``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ``sigma_bad``: Perturbed covariance matrix of shape ``(k, k)``, projected to PSD.
            - ``sigma_good``: Baseline covariance matrix of shape ``(k, k)``.

    Example:
        >>> g = torch.Generator().manual_seed(123)
        >>> sigma_bad, sigma_good = generate_sigma_bad_and_good(3, 0.5, generator=g)
        >>> sigma_bad.shape, sigma_good.shape
        (torch.Size([3, 3]), torch.Size([3, 3]))
    """
    # Step 1: Generate baseline matrices
    sigma_bad = random_vcov_matrix(k, generator=generator, var_range=var_range, device=device, dtype=dtype, eps=eps)
    sigma_good = random_vcov_matrix(k, generator=generator, var_range=var_range, device=device, dtype=dtype, eps=eps)

    # Step 2: Random mask for off-diagonal entries
    count_possible_changes = (k**2 + k) // 2 #Count diagonal entries + upper triangle
    index_change_vars = ~torch.bernoulli(torch.full((count_possible_changes,), proportion_var_dif, device=device), generator=generator).bool()

    triu_indices = torch.triu_indices(k, k, offset=0)
    indices_to_copy_sigma_bad = (triu_indices[0][index_change_vars], triu_indices[1][index_change_vars])

    sigma_good[indices_to_copy_sigma_bad] = sigma_bad[indices_to_copy_sigma_bad]
    i, j = torch.tril_indices(k, k, offset=-1)
    sigma_good[i, j] = sigma_good[j, i] # ensure symmetry
    
    sigma_good = eigen_decomp_proj_to_pd(sigma_good, eps=eps)

    return sigma_bad, sigma_good

def _mix_mean_dif_as_expected(mix_mean_dif, m, k):
    if isinstance(mix_mean_dif, float):
        return True
    if not isinstance(mix_mean_dif, torch.Tensor):
        return False
    if mix_mean_dif.dim() == 0:
        return True
    
    last_dim_compatible = mix_mean_dif.size(-1) in (k, m-1, 1)
    if not last_dim_compatible:
        return False
    
    if mix_mean_dif.dim() == 1:
        return True
    
    if mix_mean_dif.dim() > 2:
        return False
    
    return mix_mean_dif.size(0) in (m-1, 1)

def _adapt_mix_mean_dif(mix_mean_dif, m, k, security_check : bool = True, dtype : torch.dtype = torch.get_default_dtype()):
    if security_check:
        assert _mix_mean_dif_as_expected(mix_mean_dif, m, k)

    is_float = isinstance(mix_mean_dif, float)
    is_single_element_tensor = not is_float and (mix_mean_dif.numel() == 1)
    is_singleton = is_float or (mix_mean_dif.dim() == 0) or is_single_element_tensor
    if is_singleton:
        if is_float:
            mix_mean_dif = torch.tensor(mix_mean_dif, dtype = dtype)
        if is_single_element_tensor:
            mix_mean_dif = mix_mean_dif.flatten()[0]

        return mix_mean_dif.expand(m-1).unsqueeze(-1) * torch.arange(1, m).unsqueeze(-1)
    
    if mix_mean_dif.dim() == 1:
        dim_size = mix_mean_dif.size(0)
        if dim_size == k:
            return mix_mean_dif.unsqueeze(0).expand(m-1, -1) * torch.arange(1, m).unsqueeze(-1)
        if dim_size == m-1:
            return mix_mean_dif.unsqueeze(-1)
        
    if mix_mean_dif.dim() == 2:
        return mix_mean_dif

def _mix_var_dif_as_expected(mix_var_dif, m, k):
    if isinstance(mix_var_dif, float):
        return True
    if not isinstance(mix_var_dif, torch.Tensor):
        return False
    
    dim_rank = mix_var_dif.dim()
    if dim_rank == 0 or (mix_var_dif.numel() == 1):
        return True
    
    dim_rank = mix_var_dif.dim()

    if dim_rank == 1:
        return mix_var_dif.size(-1) in (m-1, 1)
    
    if dim_rank == 2:
        return mix_var_dif.shape in (torch.Size([m-1, 1]), torch.Size([k,k]))
    
    if dim_rank > 3:
        return False
    
    last_dims_ok = mix_var_dif.size(1) == mix_var_dif.size(2) == k
    return last_dims_ok and (mix_var_dif.size(0) in (m-1, 1))


def _adapt_mix_var_dif(mix_var_dif, m, k, security_check : bool = True, dtype : torch.dtype = torch.get_default_dtype()):
    if security_check:
        assert _mix_var_dif_as_expected(mix_var_dif, m, k)

    is_float = isinstance(mix_var_dif, float)
    is_single_element_tensor = not is_float and (mix_var_dif.numel() == 1)
    is_singleton = is_float or (mix_var_dif.dim() == 0) or is_single_element_tensor
    if is_singleton:
        if is_float:
            mix_var_dif = torch.tensor(mix_var_dif, dtype = dtype)
        if is_single_element_tensor:
            mix_var_dif = mix_var_dif.flatten()[0]
        return mix_var_dif#.expand(m-1).unsqueeze(-1).unsqueeze(-1)
    
    if mix_var_dif.dim() == 1:
        return mix_var_dif.unsqueeze(-1).unsqueeze(-1)
        
    if mix_var_dif.dim() == 2:
        if mix_var_dif.size(0) == k:
            return mix_var_dif.unsqueeze(0)
        if mix_var_dif.size(0) == (m-1):
            return mix_var_dif.unsqueeze(-1)
    
    if mix_var_dif.dim() == 3:
        return mix_var_dif
    

class CreditDataGenerator:
    def __init__(
            self,
            bad_mixture : GaussianMixture,
            good_mixture : GaussianMixture,
            seed : Optional[int]
    ):
        self.bad_mixture = bad_mixture
        self.good_mixture = good_mixture

        if seed is not None:
            self.bad_mixture.manual_seed(seed)

        self.good_mixture.rng = self.bad_mixture.rng

    