import torch

from typing import Dict, Optional, Tuple, Union

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

def _mix_mean_dif_as_expected(
    mix_mean_dif: Union[torch.Tensor, float],
    m: int,
    k: int,
) -> bool:
    """
    Checks whether a mean-difference specification for mixture components
    is compatible with the expected shapes.

    This helper validates that ``mix_mean_dif`` can be broadcast or adapted
    into a tensor of shape ``(m - 1, k)``, representing offsets applied to
    the base mean for each additional mixture component.

    Accepted formats:
        - A scalar float or 0-d tensor (shared scaling factor).
        - A 1D tensor of length ``k`` (per-covariate offsets).
        - A 1D tensor of length ``m - 1`` (per-component scaling).
        - A 2D tensor of shape ``(m - 1, k)``.
        - Singleton dimensions (size 1) are allowed and broadcastable.

    Args:
        mix_mean_dif (Tensor or float):
            Mean difference specification.
        m (int):
            Number of mixture components.
        k (int):
            Number of covariates (feature dimension).

    Returns:
        bool:
            ``True`` if the input is structurally compatible, ``False`` otherwise.
    """

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

def _adapt_mix_mean_dif(
    mix_mean_dif: Union[torch.Tensor, float],
    m: int,
    k: int,
    security_check: bool = True,
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    """
    Adapts a mean-difference specification into a tensor of shape ``(m - 1, k)``.

    This function normalizes various user-friendly input formats into a
    canonical tensor representation suitable for constructing mixture means.
    The first component is assumed to be the base mean; the returned tensor
    contains offsets for the remaining ``m - 1`` components.

    Broadcasting rules:
        - Scalars are expanded linearly with component index.
        - Per-covariate vectors are shared across components.
        - Per-component vectors are expanded across covariates.
        - 2D tensors are assumed to already be in canonical form.

    Args:
        mix_mean_dif (Tensor or float):
            Mean difference specification.
        m (int):
            Number of mixture components.
        k (int):
            Number of covariates.
        security_check (bool, default=True):
            If ``True``, validates the input shape before adaptation.
        dtype (torch.dtype, default=torch.get_default_dtype()):
            Target dtype when constructing tensors from Python scalars.

    Returns:
        Tensor:
            A tensor of shape ``(m - 1, k)`` or a compatible shape.
    """
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

def _mix_var_dif_as_expected(
    mix_var_dif: Union[torch.Tensor, float],
    m: int,
    k: int,
) -> bool:
    """
    Checks whether a variance-difference specification for mixture components
    is compatible with the expected shapes. It **does not** check for positive
    definitness or symmetry.

    The variance difference is expected to represent adjustments to covariance
    matrices for mixture components beyond the base component.

    Accepted formats:
        - Scalar float or singleton tensor.
        - 1D tensor of length ``m - 1`` (per-component scaling).
        - 2D tensor of shape ``(k, k)`` or ``(m - 1, 1)``.
        - 3D tensor of shape ``(m - 1, k, k)`` or ``(1, k, k)``.

    Args:
        mix_var_dif (Tensor or float):
            Variance difference specification.
        m (int):
            Number of mixture components.
        k (int):
            Number of covariates.

    Returns:
        bool:
            ``True`` if the input is structurally compatible, ``False`` otherwise.
    """

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


def _adapt_mix_var_dif(
    mix_var_dif: Union[torch.Tensor, float],
    m: int,
    k: int,
    security_check: bool = True,
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    """
    Adapts a variance-difference specification into a canonical tensor form.

    The returned tensor represents covariance adjustments for mixture
    components beyond the base component.

    Broadcasting rules:
        - Scalars are treated as uniform adjustments.
        - 1D tensors are expanded to diagonal covariance adjustments.
        - 2D tensors are interpreted as shared covariance matrices.
        - 3D tensors are assumed to already be in canonical form.

    Args:
        mix_var_dif (Tensor or float):
            Variance difference specification.
        m (int):
            Number of mixture components.
        k (int):
            Number of covariates.
        security_check (bool, default=True):
            If ``True``, validates the input shape before adaptation.
        dtype (torch.dtype, default=torch.get_default_dtype()):
            Target dtype when constructing tensors from Python scalars.

    Returns:
        Tensor:
            A tensor representing variance adjustments with shape compatible
            with ``(m - 1, k, k)`` broadcasting.
    """
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
        return mix_var_dif
    
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

    bad_good_encoding = {
        "bad" : 1,
        "good" : 0
    }

    def __init__(
            self,
            bad_mixture : GaussianMixture,
            good_mixture : GaussianMixture,
            noise_var : float,
            bad_ratio : float,
            seed : Optional[int] = None
    ):
        if not isinstance(bad_mixture, GaussianMixture) and not isinstance(good_mixture, GaussianMixture):
            raise ValueError("Mixtures need to be GaussianMixture classes")
        
        self.bad_mixture = bad_mixture
        self.good_mixture = good_mixture
        self.noise_std = torch.sqrt(torch.tensor(float(noise_var)))

        if seed is not None:
            self.bad_mixture.manual_seed(seed)

        self.good_mixture.rng = self.bad_mixture.rng

    @property
    def device(self) -> torch.device:
        """
        The device on which both the good and bad Gaussian mixture parameters reside.
        """
        return self.bad_mixture.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.bad_mixture.mean.dtype
    
    @property
    def rng(self) -> torch.Generator:
        return self.bad_mixture.rng
    
    
    def to(
            self, device : torch.device, seed : Optional[int] = None, set_same_initial_seed : bool = True
    ):
        """
        Moves the internal Gaussian mixture generators to the specified device.

        Both the good and bad mixtures are transferred to the target device.
        A shared random number generator is recreated to ensure consistent
        sampling behavior across mixtures.

        Args:
            device (torch.device):
                Target device.
            seed (int, optional):
                Explicit seed for the shared random number generator.
            set_same_initial_seed (bool, default=True):
                If ``True`` and ``seed`` is ``None``, preserves the original
                initial seed when recreating the generator.

        Returns:
            CreditDataGenerator:
                The current instance moved to the specified device.
        """
        self.bad_mixture.to(device, seed, set_same_initial_seed)
        self.good_mixture.to(device, seed, set_same_initial_seed)
        self.good_mixture.rng = self.bad_mixture.rng
        return self
    
    def sample(
            self, 
            n : int, 
            deterministic_weights_for_mixture_sampling : bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_bad = round(self.bad_ratio * n)
        n_good = round((1-self.bad_ratio) * n)
        if (n_bad + n_good) != n:
            adapt_n_bad = torch.randint(low=0,high=2,size=(1,),generator=self.rng).to(bool).item()
            if adapt_n_bad:
                n_bad = n - n_good
            else:
                n_good = n - n_bad

        dtype = self.dtype
        device = self.device

        X_bad = self.bad_mixture.sample(n_bad, deterministic_weights = deterministic_weights_for_mixture_sampling) # [n, k]
        y_bad = torch.full((n_bad,), self.bad_good_encoding["bad"], device=device, dtype=dtype) # [n]

        X_good = self.bad_mixture.sample(n_bad, deterministic_weights = deterministic_weights_for_mixture_sampling) # [n, k]
        y_good = torch.full((n_good,), self.bad_good_encoding["good"], device=device, dtype=dtype) # [n]

        X = torch.cat([X_bad, X_good], dim=0)
        y = torch.cat([y_bad, y_good])

        X = X + torch.randn(X.shape, generator=self.rng, device=device, dtype=dtype) / self.noise_std

        return X, y

    
    @staticmethod
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
            >>> sigma_bad, sigma_good = CreditDataGenerator.generate_sigma_bad_and_good(3, 0.5, generator=g)
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

    @classmethod
    def init_with_internal_logic(
        cls,
        count_covariates : int = 10,
        mean_bad_diff : Union[torch.Tensor, float] = 1.0,
        con_var_bad_dif : float = 0.0,
        covars :  Optional[Dict[str, torch.Tensor]]  = None, # keys = ["bad", "good"]
        iid : bool              = False,
        mixture_weights  : Optional[torch.Tensor]    = None,
        mix_mean_dif_bad  : Union[torch.Tensor, float]   = None,
        mix_mean_dif_good  : Union[torch.Tensor, float]   = None,
        mix_var_dif_bad  : Union[torch.Tensor, float]   = None,
        mix_var_dif_good  : Union[torch.Tensor, float]   = None,
        noise_var : float = 0.1,
        bad_ratio : float = 0.5,
        device : Optional[torch.device] = None, 
        dtype : torch.dtype = torch.get_default_dtype(),
        do_security_checks : bool = True,
        seed_var_gen : Optional[int] = None,
        seed_credit_data_gen : Optional[int] = None
    ):
        """
        Constructs a ``CreditDataGenerator`` using internally generated mixture
        parameters following predefined structural rules.

        This factory method supports IID or correlated covariates, optional
        mixture structures, and controlled mean/variance offsets between good
        and bad populations.

        Args:
            count_covariates (int, default=10):
                Number of covariates.
            mean_bad_diff (Tensor or float, default=1.0):
                Mean shift applied to the good population.
            con_var_bad_dif (float, default=0.0):
                Proportional variance difference between populations.
            covars (dict, optional):
                Explicit covariance matrices with keys ``"bad"`` and ``"good"``.
            iid (bool, default=False):
                If ``True``, uses identity covariance matrices.
            mixture_weights (Tensor, optional):
                Mixture weights, shape ``(m,)`` or ``(2, m)``.
            mix_mean_dif_bad, mix_mean_dif_good (Tensor or float, optional):
                Mean offsets for mixture components.
            mix_var_dif_bad, mix_var_dif_good (Tensor or float, optional):
                Variance offsets for mixture components.
            device (torch.device, optional):
                Target device.
            dtype (torch.dtype, default=torch.get_default_dtype()):
                Target dtype.
            do_security_checks (bool, default=True):
                Enables input validation.
            seed_var_gen (int, optional):
                Seed for covariance generation.
            seed_credit_data_gen (int, optional):
                Seed for the credit data generator RNG.

        Returns:
            CreditDataGenerator:
                A fully initialized credit data generator.
        """
        # 0. Process "None" logic path and ensure everything is well set
        ## ensure device is defined
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs_for_generated_tensors = {"dtype" : dtype, "device" : device}

        mu_bad = torch.zeros(count_covariates, **kwargs_for_generated_tensors)
        mu_good = mu_bad + mean_bad_diff

        if iid:
            sigma_bad, sigma_good = [torch.eye(count_covariates, **kwargs_for_generated_tensors) for _ in range(2)]
            mix_var_dif_bad, mix_var_dif_good = 0.0, 0.0
        elif covars is not None:
            sigma_bad = covars["bad"]
            sigma_good = covars["good"]
        else:
            rng = torch.Generator(device=device)
            if seed_var_gen is not None:
                rng.manual_seed(1807)
            sigma_bad, sigma_good = cls.generate_sigma_bad_and_good(
                k = count_covariates, 
                proportion_var_dif=con_var_bad_dif, 
                generator = rng, 
                device=device,
                dtype= dtype
            )

        if mixture_weights is None:
            weights_bad = None
            weights_good = None
        else:
            if mixture_weights.dim() == 1:
                weights_bad = mixture_weights
                weights_good = mixture_weights.copy()
            else:
                if do_security_checks:
                    assert mixture_weights.dim() == 2, "mixture_weights has to be a tensor of dim in (1,2)"
                    assert mixture_weights.size(0) == 2, "Shape should be [2, m]"
                weights_bad = mixture_weights[0]
                weights_good = mixture_weights[1]
            
            m = weights_bad.size(-1)

            mix_mean_dif_bad, mix_mean_dif_good = [_adapt_mix_mean_dif(d, m, k=count_covariates, security_check=do_security_checks) 
                                                   for d in (mix_mean_dif_bad, mix_mean_dif_good)]
            
            amplify_base_with_dif = lambda param, dif : torch.cat([param.unsqueeze(0), param.unsqueeze(0) + dif],
                                                                  dim=0)

            mu_bad, mu_good = [amplify_base_with_dif(mu, dif)
                               for mu, dif in [(mu_bad, mix_mean_dif_bad), (mu_good, mix_mean_dif_good)]]
            
            sigma_bad, sigma_good = [_adapt_mix_var_dif(d, m, k=count_covariates, security_check=do_security_checks) 
                                                    for d in (mix_var_dif_bad, mix_var_dif_good)]
                
        mixture_bad = GaussianMixture(
            mean = mu_bad,
            cov = sigma_bad,
            weights = weights_bad,
            seed = None,
            cov_symmetry_rtol_atol = [0.0, 0.0],
            check_params = True
        )
        
        mixture_good = GaussianMixture(
            mean = mu_good,
            cov = sigma_good,
            weights = weights_good,
            seed = None,
            cov_symmetry_rtol_atol = [0.0, 0.0],
            check_params = True
        )

        return cls(
            bad_mixture = mixture_bad,
            good_mixture = mixture_good,
            seed = seed_credit_data_gen,
            noise_var=noise_var
        )