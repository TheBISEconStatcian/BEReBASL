from torch.utils.data import Dataset
import torch


from typing import Any, Dict, Literal, Optional, Tuple, Union

from berebasl.simulation.gaussian_mixture import (
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
    dtype: torch.dtype = None,
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
        dtype (torch.dtype, default=None):
            Target dtype when constructing tensors from Python scalars. When None
            the value is ``torch.get_default_dtype()``

    Returns:
        Tensor:
            A tensor of shape ``(m - 1, k)`` or a compatible shape.
    """
    if dtype is None:
            dtype = torch.get_default_dtype()
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
    dtype: torch.dtype = None,
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
            Target dtype when constructing tensors from Python scalars. When None
            the value is ``torch.get_default_dtype()``

    Returns:
        Tensor:
            A tensor representing variance adjustments with shape compatible
            with ``(m - 1, k, k)`` broadcasting.
    """
    if dtype is None:
            dtype = torch.get_default_dtype()
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
        self.bad_ratio = float(bad_ratio)
        
        self.add_noise = self.noise_std > 0

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
    
    def manual_seed(self, seed : int) -> torch.Generator:
        self.rng.manual_seed(seed)
    
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

        if self.add_noise:
            X = X + torch.randn(X.shape, generator=self.rng, device=device, dtype=dtype) * self.noise_std

        return X, y

    
    @staticmethod
    def generate_sigma_bad_and_good(
        k: int,
        proportion_var_dif: float,
        generator: torch.Generator,
        var_range: Tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-6,
        device: torch.device = torch.device("cpu"),
        dtype: Optional[torch.dtype] = None
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
            device (torch.device, optional): Device for tensor allocation. Defaults to CPU. When None
                the value is ``torch.get_default_dtype()``
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
        if dtype is None:
            dtype = torch.get_default_dtype()
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
        dtype : Optional[torch.dtype] = None,
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
            noise_var (float, default=0.1):
                Variance of the 0-mean normally distributed noise to be added to covariate
                samples.
            bad_ratio (float, default=0.5):
                ratio of bad among total ``n`` per sample.
            device (torch.device, optional):
                Target device.
            dtype (torch.dtype, default=None):
                Target dtype. When None it is set to torch.get_default_dtype()
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
        if dtype is None:
            dtype = torch.get_default_dtype()

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
            noise_var=noise_var,
            bad_ratio=bad_ratio
        )
    


class CreditDataSample(Dataset):
    """Leakage-safe dataset for reject inference experiments.

    This class provides a structured and safe representation of credit data
    split into *accepted* and *rejected* applications. It enforces the core
    reject-inference constraint that rejected applications must never expose
    their repayment outcomes, while accepted applications retain both features
    and labels.

    The dataset supports:
    - retrieval of either accepted or rejected samples
    - reproducible train/test splits
    - device transfers with preservation of RNG state
    - manual reseeding for deterministic behavior

    A device-specific :class:`torch.Generator` instance is maintained internally
    and is used for all stochastic operations, including permutation and
    train/test splitting. The generator is transferred when calling :meth:`to`
    and can be reseeded via :meth:`manual_seed` to ensure reproducibility across
    runs and devices.

    Attributes:
        features_rejects (torch.Tensor):
            Feature matrix for rejected applications. Shape ``(n_rejects, n_features)``.
        features_accepts (torch.Tensor):
            Feature matrix for accepted applications. Shape ``(n_accepts, n_features)``.
        default_flag_accepts (torch.Tensor):
            Repayment outcomes for accepted applications. Shape ``(n_accepts,)``.
        retrieve_only_accepted (bool):
            If ``True``, the dataset yields only accepted samples. If ``False``,
            only rejected samples are returned.
        rng (torch.Generator):
            Device-specific random number generator used for all stochastic
            operations.
    """

    def __init__(
        self,
        features_rejects: torch.Tensor,
        features_accepts: torch.Tensor,
        default_flag_accepts: torch.Tensor,
        retrieve_only_accepted: bool,
        seed: Optional[int] = None,
    ):
        """Initialize a leakage-safe credit dataset sample.

        Args:
            features_rejects (torch.Tensor):
                Feature matrix for rejected applications. Must reside on the same
                device as the other tensors. Shape ``(n_rejects, n_features)``.
            features_accepts (torch.Tensor):
                Feature matrix for accepted applications. Shape ``(n_accepts, n_features)``.
            default_flag_accepts (torch.Tensor):
                Repayment outcomes for accepted applications. Shape ``(n_accepts,)``.
            retrieve_only_accepted (bool):
                If ``True``, the dataset yields only accepted samples during
                indexing. If ``False``, only rejected samples are returned.
            seed (int, optional):
                Seed used to initialize the internal device-specific random number
                generator. If ``None``, the generator is left in its default state.

        Notes:
            - Rejected samples never expose repayment outcomes.
            - The internal RNG controls all stochastic behavior, including
              train/test splitting and permutations.
            - The RNG is device-specific and is recreated when calling :meth:`to`.
        """
        self.features_rejects = features_rejects
        self.features_accepts = features_accepts
        self.default_flag_accepts = default_flag_accepts

        self.retrieve_only_accepted = retrieve_only_accepted

        # RNG is device-specific, so we create it on the same device as the data
        device = features_rejects.device
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

    # -------------------------------------------------------------------------
    # RNG utilities
    # -------------------------------------------------------------------------

    def manual_seed(self, seed: int):
        """Manually reseed the internal random number generator.

        Args:
            seed (int): New seed value.

        Returns:
            CreditDataSample: The dataset instance.
        """
        self.rng.manual_seed(seed)
        return self

    # -------------------------------------------------------------------------
    # Device transfer
    # -------------------------------------------------------------------------

    def to(
        self,
        device: torch.device,
        seed: Optional[int] = None,
        set_same_initial_seed: bool = True,
    ):
        """Move all tensors and the RNG to a target device.

        Args:
            device (torch.device):
                Target device for the dataset tensors.
            seed (int, optional):
                Explicit seed for the RNG on the new device. Overrides
                ``set_same_initial_seed`` if provided.
            set_same_initial_seed (bool, optional):
                If ``True`` and ``seed`` is ``None``, the RNG on the new device
                is initialized with the same initial seed as the previous RNG.

        Returns:
            CreditDataSample: The dataset instance with tensors moved to ``device``.
        """
        for var in ["features_rejects", "features_accepts", "default_flag_accepts"]:
            setattr(self, var, getattr(self, var).to(device))

        # Preserve reproducibility across device transfers
        if seed is None and set_same_initial_seed:
            seed = self.rng.initial_seed()

        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

        return self

    # -------------------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------------------

    def _generate_test_mask(
        self,
        n: int,
        test_proportion: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a boolean mask selecting test samples.

        Args:
            n (int): Number of samples.
            test_proportion (float): Fraction of samples to assign to the test set.
            device (torch.device): Device for the mask.

        Returns:
            torch.Tensor: Boolean mask of shape ``(n,)``.
        """
        test_count = round(test_proportion * n)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        perm = torch.randperm(n, generator=self.rng, device=device)
        mask[perm[:test_count]] = True
        return mask

    def train_test_split(self, test_proportion: float):
        """Split the dataset into train and test subsets.

        The split is performed independently for accepts and rejects, ensuring
        no leakage and preserving the acceptance structure.

        Args:
            test_proportion (float):
                Fraction of samples to assign to the test set. Must be in ``[0, 1]``.

        Returns:
            Tuple[CreditDataSample, CreditDataSample]:
                ``(train_sample, test_sample)``.
        """
        if not (0.0 <= test_proportion <= 1.0):
            raise ValueError("test_proportion must be between 0 and 1")

        device = self.features_rejects.device

        # Generate masks
        test_mask_rej = self._generate_test_mask(
            self.features_rejects.size(0), test_proportion, device
        )
        test_mask_acc = self._generate_test_mask(
            self.features_accepts.size(0), test_proportion, device
        )

        # Slice data
        train_sample = CreditDataSample(
            features_rejects=self.features_rejects[~test_mask_rej],
            features_accepts=self.features_accepts[~test_mask_acc],
            default_flag_accepts=self.default_flag_accepts[~test_mask_acc],
            retrieve_only_accepted=self.retrieve_only_accepted,
            seed=self.rng.initial_seed(),
        )

        test_sample = CreditDataSample(
            features_rejects=self.features_rejects[test_mask_rej],
            features_accepts=self.features_accepts[test_mask_acc],
            default_flag_accepts=self.default_flag_accepts[test_mask_acc],
            retrieve_only_accepted=self.retrieve_only_accepted,
            seed=self.rng.initial_seed(),
        )

        return train_sample, test_sample

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    @property
    def count_accepts(self) -> int:
        """Number of accepted samples."""
        return self.features_accepts.size(0)

    @property
    def count_rejects(self) -> int:
        """Number of rejected samples."""
        return self.features_rejects.size(0)

    def __len__(self) -> int:
        """Dataset length under the current retrieval mode."""
        return self.count_accepts if self.retrieve_only_accepted else self.count_rejects

    def __getitem__(self, idx: int) -> Tuple[Any, Optional[torch.Tensor]]:
        """Retrieve a single sample.

        Returns:
            Tuple[Any, Optional[torch.Tensor]]:
                - If retrieving accepts: ``(features, default_flag)``
                - If retrieving rejects: ``(features, None)``
        """
        if self.retrieve_only_accepted:
            return (
                self.features_accepts[idx],
                self.default_flag_accepts[idx],
            )
        else:
            return (
                self.features_rejects[idx],
                None,
            )




class CreditData(Dataset):
    """Dataset for credit rating simulation with reject inference.

    This dataset stores applicant features, default flags (binary repayment outcome),
    acceptance indicators, and per-sample generation round identifiers. It supports
    retrieval of either the full dataset, only accepted applications, or only
    rejected applications, controlled via a retrieval mode. The class is designed
    for dynamically expanding the observed data per generation round.

    Args:
        features_initial (torch.Tensor):
            Shape ``(n_samples, n_features)``. Applicant feature matrix.
        default_flag_initial (torch.Tensor):
            Shape ``(n_samples,)``. Binary repayment outcome where ``0`` = repaid
            and ``1`` = default. Values are expected to be in ``{0, 1}``.
        accepted_initial (torch.Tensor):
            Shape ``(n_samples,)``. Boolean acceptance status of applications.
        retrieval_mode (Literal["accepts", "rejects", "unbiased"], optional):
            Retrieval mode for indexing. ``"accepts"`` yields only accepted
            applications, ``"rejects"`` yields only rejected applications, and
            ``"unbiased"`` yields all observations. Defaults to ``"accepts"``.

    Raises:
        ValueError:
            If input tensors are on different devices, have incompatible shapes,
            if ``accepted_initial`` is not one-dimensional, or if
            ``retrieval_mode`` is not recognized.

    Attributes:
        features (torch.Tensor): Applicant features across generations.
        default_flag (torch.Tensor): Repayment outcomes across generations.
        accepted (torch.Tensor): Boolean acceptance flags per sample.
        accepted_idx (torch.Tensor): Indices of accepted applications (1D).
        reject_idx (torch.Tensor): Indices of rejected applications (1D).
        gen_round (torch.Tensor): Generation round index for each sample.
        retrieval_mode (str): Current retrieval mode.
    """
    def __init__(
            self, 
            features_initial : torch.Tensor, 
            default_flag_initial : torch.Tensor, 
            accepted_initial : torch.Tensor,
            retrieval_mode : Literal["accepts", "rejects", "unbiased"] = "accepts"
    ):
        if not (features_initial.device == default_flag_initial.device == accepted_initial.device):
            raise ValueError("Not all args have the same device")
        
        if not (features_initial.size(0) == default_flag_initial.size(0) == accepted_initial.size(0)):
            raise ValueError("Shapes are non-compatible")
        
        if accepted_initial.dim() != 1:
            raise ValueError("accepted_initial needs to be one dimensional")

        if retrieval_mode not in ("accepts", "rejects", "unbiased"):
            raise ValueError('retrieval_mode must be one of "accepts", "rejects", "unbiased"')

        self.features = features_initial.detach().clone()
        self.default_flag = default_flag_initial.detach().clone().to(self.features.dtype)
        self.accepted = accepted_initial.detach().clone().to(bool)

        # Cache indices for accepts and rejects as 1D tensors
        self.accepted_idx = torch.nonzero(self.accepted).flatten()
        self.reject_idx = torch.nonzero(~self.accepted).flatten()

        self.gen_round = torch.tensor(
            0, dtype=torch.long, device=features_initial.device
        ).expand(features_initial.size(0))

        self.retrieval_mode = retrieval_mode

    # -------------------------------------------------------------------------
    # Device handling
    # -------------------------------------------------------------------------

    def to(self, device: torch.device):
        """Move all internal tensors to a target device.

        Args:
            device (torch.device):
                Target device for the dataset tensors (e.g., ``torch.device('cuda')``).

        Returns:
            CreditData: The dataset instance with tensors moved to ``device``.
        """
        for var in ["features", "default_flag", "accepted_idx", "reject_idx", "accepted", "gen_round"]:
            setattr(self, var, getattr(self, var).to(device))

        return self

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device on which the dataset tensors currently reside.

        Returns:
            torch.device: The device of ``features`` (and thus all dataset tensors).
        """
        return self.features.device
    
    @property
    def last_gen_round(self) -> torch.Tensor:
        """Last generation round identifier/index.

        This corresponds to the maximum generation round identifier present in
        the dataset

        Returns:
            torch.Tensor:
                Scalar long tensor indicating the last generation round.
        """
        return self.gen_round[-1]
    
    @property
    def count_accepts(self) -> int:
        """Number of accepted applications in the dataset.

        Returns:
            int:
                Count of samples where the acceptance flag is ``True``.
        """
        return self.accepted_idx.size(0)
    
    @property
    def count_all(self) -> int:
        """Total number of observations in the dataset.

        Returns:
            int:
                Total number of samples stored in the dataset.
        """
        return self.accepted.size(0)
    
    @property
    def count_rejects(self) -> int:
        """Number of rejected applications in the dataset.

        Returns:
            int:
                Count of samples where the acceptance flag is ``False``.
        """
        return self.count_all - self.count_accepts

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def add_gen(
            self, 
            features_new : torch.Tensor,
            default_flag_new : torch.Tensor, 
            accepted_new : torch.Tensor
        ) -> None:
        """Append a new generation of samples to the dataset.

        Concatenates new features, outcomes, and acceptance flags, updates
        accepted and rejected indices, and assigns the next generation round 
        identifier to the appended samples.

        Args:
            features_new (torch.Tensor):
                Shape ``(m_samples, n_features)``. New applicant features.
                Must match ``features.shape[1]``.
            default_flag_new (torch.Tensor):
                Shape ``(m_samples,)``. Binary repayment outcomes in ``{0, 1}``.
            accepted_new (torch.Tensor):
                Shape ``(m_samples,)``. Boolean acceptance flags.

        Raises:
            ValueError: If feature dimensionality of ``features_new`` does not match
                existing ``features``.
        """
        if features_new.size(1) != self.features.size(1):
            raise ValueError("Feature dimension mismatch in features_new")
        
        self.features = torch.cat([self.features, features_new])
        self.default_flag = torch.cat([self.default_flag, default_flag_new.to(self.features.dtype)])

        obs_count_before_adding_new_gen = self.count_all

        # Update accepted and rejected indices
        new_accepted_idx = torch.nonzero(accepted_new).flatten() + obs_count_before_adding_new_gen
        new_reject_idx = torch.nonzero(~accepted_new).flatten() + obs_count_before_adding_new_gen

        self.accepted_idx = torch.cat([self.accepted_idx, new_accepted_idx])
        self.reject_idx = torch.cat([self.reject_idx, new_reject_idx])

        self.accepted = torch.cat([self.accepted, accepted_new])

        new_gen_round = self.last_gen_round + 1
        self.gen_round = torch.cat(
            [self.gen_round, new_gen_round.expand(features_new.size(0))]
        )

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def rejects(
        self, include_gen_round: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return feature observations corresponding to rejected applications.

        This method provides access to the rejected samples without exposing their
        repayment outcomes, ensuring no label leakage during reject inference
        experiments. Optionally, the generation round of each rejected sample can
        also be returned.

        Args:
            include_gen_round (bool, optional):
                If ``True``, returns both ``features`` and ``gen_round`` for rejected
                samples. If ``False``, returns only ``features``.
                Defaults to ``False``.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If ``include_gen_round=False``: ``features_rejects``  
                - If ``include_gen_round=True``: ``(features_rejects, gen_round_rejects)``
        """
        if include_gen_round:
            return self.features[self.reject_idx], self.gen_round[self.reject_idx]
        return self.features[self.reject_idx]

    def accepts(
        self, include_gen_round: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return observations corresponding to accepted applications.

        Accepted samples include both features and repayment outcomes. The generation
        round can optionally be included.

        Args:
            include_gen_round (bool, optional):
                If ``True``, returns ``(features, default_flag, gen_round)`` for accepted
                samples. If ``False``, returns ``(features, default_flag)``.
                Defaults to ``False``.

        Returns:
            Tuple[torch.Tensor, ...]:
                - If ``include_gen_round=False``: ``(features_accepts, default_flag_accepts)``
                - If ``include_gen_round=True``: ``(features_accepts, default_flag_accepts, gen_round_accepts)``
        """
        if include_gen_round:
            return (
                self.features[self.accepted_idx],
                self.default_flag[self.accepted_idx],
                self.gen_round[self.accepted_idx],
            )
        return self.features[self.accepted_idx], self.default_flag[self.accepted_idx]

    def unbiased_obs(
        self, include_gen_round: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full dataset without any acceptance-based filtering.

        This method exposes all observations exactly as stored, making it suitable
        for analyses that require the complete underlying sample distribution.

        Args:
            include_gen_round (bool, optional):
                If ``True``, returns ``(features, default_flag, gen_round)``.
                If ``False``, returns ``(features, default_flag)``.
                Defaults to ``False``.

        Returns:
            Tuple[torch.Tensor, ...]:
                - If ``include_gen_round=False``: ``(features, default_flag)``
                - If ``include_gen_round=True``: ``(features, default_flag, gen_round)``
        """
        if include_gen_round:
            return self.features, self.default_flag, self.gen_round
        return self.features, self.default_flag

    def to_sample_dataset(
        self,
        retrieve_only_accepted: bool = True,
    ) -> CreditDataSample:
        """Create a leakage-safe sample dataset from current observations.

        This method constructs a :class:`CreditDataSample` instance from the
        dataset's present observations. Rejected applications contribute only
        their feature vectors, while accepted applications contribute both
        features and repayment outcomes. No labels from rejected samples are
        ever exposed, ensuring reject-inference safety.

        Args:
            retrieve_only_accepted (bool, optional):
                If ``True``, the returned :class:`CreditDataSample` will yield only
                accepted samples during indexing. If ``False``, only rejected
                samples will be returned. Defaults to ``True``.

        Returns:
            CreditDataSample:
                A leakage-safe dataset containing the current accepted and rejected
                observations, suitable for training, evaluation, or splitting.
        """
        return CreditDataSample(
            self.rejects(include_gen_round=False),
            *self.accepts(include_gen_round=False),
            retrieve_only_accepted=retrieve_only_accepted,
        )

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of samples available under the current retrieval policy.

        Returns:
            int:
                - If ``retrieval_mode == "accepts"``: number of accepted samples.
                - If ``retrieval_mode == "rejects"``: number of rejected samples.
                - If ``retrieval_mode == "unbiased"``: total number of samples.
        """
        if self.retrieval_mode == "accepts":
            return self.count_accepts
        elif self.retrieval_mode == "rejects":
            return self.count_rejects
        elif self.retrieval_mode == "unbiased":
            return self.count_all
        else:
            raise ValueError("saved retrieval_mode not recognized")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a single sample as a dictionary.

        Indexing respects the current ``retrieval_mode``:
        - ``"accepts"``: ``idx`` is mapped through ``accepted_idx``.
        - ``"rejects"``: ``idx`` is mapped through ``reject_idx``.
        - ``"unbiased"``: ``idx`` addresses the full dataset.

        Args:
            idx (int):
                Sample index under the current retrieval mode.

        Returns:
            Dict[str, Any]:
                A dictionary with keys:
                    - ``"features"`` (torch.Tensor): Feature vector.
                    - ``"default_flag"`` (torch.Tensor): Repayment outcome.
                    - ``"accepted"`` (torch.Tensor): Boolean acceptance flag.
                    - ``"gen_round"`` (torch.Tensor): Generation round index.
        """
        if self.retrieval_mode == "accepts":
            idx = self.accepted_idx[idx]
        elif self.retrieval_mode == "rejects":
            idx = self.reject_idx[idx]
        elif self.retrieval_mode == "unbiased":
            pass
        else:
            raise ValueError("saved retrieval_mode not recognized")

        features = self.features[idx]
        default_flag = self.default_flag[idx]
        accepted = self.accepted[idx]
        gen_round = self.gen_round[idx]

        return {
            "features": features,
            "default_flag": default_flag,
            "accepted": accepted,
            "gen_round": gen_round,
        }

    def data_stats(self) -> Dict[str, Union[float, int]]:
        """Compute descriptive statistics of the current dataset.

        The statistics are based on the acceptance flags and default outcomes and
        are computed across all samples currently stored in the dataset.

        Returns:
            Dict[str, Union[float, int]]:
                Dictionary containing:
                    - ``sample_size``: Total number of observations.
                    - ``accept_ratio``: Fraction of accepted applications among all observations.
                    - ``bad_ratio_accepts``: Default rate among accepted applications.
                    - ``bad_ratio_rejects``: Default rate among rejected applications.
                    - ``bad_ratio_unbiased``: Overall default rate across all observations.

        Raises:
            ZeroDivisionError:
                If there are no accepted and no rejected applications when computing
                the corresponding unbiased bad rate.
        """
        bads_count_among_accepts = self.default_flag[self.accepted_idx].sum().item()
        bads_count_among_rejects = self.default_flag[~self.accepted].sum().item()
        all_obs_count = self.count_all

        stats = {
            "sample_size" : all_obs_count,
            "accept_ratio" : self.count_accepts / all_obs_count,
            "bad_ratio_accepts" : float("nan") if self.count_accepts == 0 else bads_count_among_accepts / self.count_accepts,
            "bad_ratio_rejects" : float("nan") if self.count_rejects == 0 else bads_count_among_rejects / self.count_rejects,
            "bad_ratio_unbiased" : (bads_count_among_accepts + bads_count_among_rejects) / all_obs_count # assumes there is at least one reject or accept
        }
        
        return stats


def accept_based_on_top_percentent_of_arbitrary_var(
        features : torch.Tensor, 
        default_flag : torch.Tensor, 
        var_for_rule : int,
        top_percent : float,
        default_value : int = 1, # 1 or 0
        min_count_bads : int = 4
):
    if var_for_rule >= features.shape[1]:
        raise ValueError("var_for_rule outside of index")
    
    cutoff = torch.quantile(features[:, var_for_rule], 1 - top_percent)

    accepts = features[:, var_for_rule] >= cutoff


    count_defaults_within_accepts = (default_flag[accepts] == default_value).sum()
    if count_defaults_within_accepts < min_count_bads:
        lidx_defaults_non_accepted = (~accepts) & (default_flag == default_value)
        defaults_still_selectable = lidx_defaults_non_accepted.sum()
        if defaults_still_selectable == 0:
            return accepts
        
        count_bads_to_still_achieve = min_count_bads -count_defaults_within_accepts
        var_for_rule_vals_of_rejected_defaults = features[lidx_defaults_non_accepted][:, var_for_rule]
        
        if defaults_still_selectable <= count_bads_to_still_achieve:
            var_for_rule_vals_of_rejected_defaults = features[lidx_defaults_non_accepted][:, var_for_rule]
            accept_rule_to_include_all_defaults = features[:, var_for_rule] >=  var_for_rule_vals_of_rejected_defaults.min()
            return accept_rule_to_include_all_defaults
        

        new_cutoff = torch.topk(var_for_rule_vals_of_rejected_defaults,k=count_bads_to_still_achieve, largest=True).values[-1]

        return features[:, var_for_rule] >= new_cutoff
    
    return accepts

if __name__ == "__main__":
    print('*' * 10,"Initializing credit data simulation", '*' * 10)

