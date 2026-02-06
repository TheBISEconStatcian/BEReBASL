##### Base libraries
from copy import deepcopy
from types import FunctionType
from math import isnan
from warnings import warn

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

##### Third Party libraries
from torch.utils.data import Dataset
import torch

##### Internal imports
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

        X_good = self.good_mixture.sample(n_good, deterministic_weights = deterministic_weights_for_mixture_sampling) # [n, k]
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
    
def _mask2d_to_int_idxs(mask : torch.Tensor, correction_last_idx : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_int = mask.to(torch.int32)
    idx_last_axis = torch.where(# [B, N]
            mask,
            mask_int.cumsum(-1) - 1,
            -1
    )
    if correction_last_idx is not None:
        idx_last_axis = idx_last_axis + correction_last_idx
    batch_idx = torch.arange(mask.size(0), device=mask.device).unsqueeze(-1).expand(mask.shape)
    
    return batch_idx[mask], idx_last_axis[mask]

class CreditDataSample(Dataset):
    r"""
    Leakage-safe, super-batch-aware dataset for reject-inference experiments.

    This class represents credit application data split into *accepted* (labeled)
    and *rejected* (unlabeled) groups, enforcing the core reject-inference rule:
    rejected applications never expose repayment outcomes. Accepted applications
    retain both features and labels.

    All tensors support arbitrary super-batch shapes. The final two dimensions
    follow the contract:

        ``features_unlabeled``:  ``(..., N_unlabeled, F)``
        ``features_labeled``:    ``(..., N_labeled,   F)``
        ``labels``:              ``(..., N_labeled)``
        ``_unlabeled_ids``:      ``(..., N_unlabeled)``
        ``_ids_inferred``:       ``(..., N_labeled)``

    where the leading dimensions represent super-batches.

    Key features:
        - leakage-safe retrieval of accepted or rejected samples
        - super-batch-compatible train/test splits using gather-based indexing
        - device-specific RNG with reproducible behavior across transfers
        - deterministic reseeding via :meth:`manual_seed`
        - configurable NaN encodings for labels and IDs
        - shape-agnostic reject-labeling workflow that:
            * appends inferred labels into the labeled pool
            * compacts the unlabeled pool
            * preserves ID alignment
            * avoids Python loops entirely

    Attributes:
        features_unlabeled (Tensor): Rejected application features.
        features_labeled (Tensor): Accepted application features.
        labels (Tensor): Repayment outcomes for accepted applications.
        retrieve_only_labeled (bool): Whether ``__getitem__`` returns labeled samples.
        rng (torch.Generator): Device-specific RNG for all stochastic operations.
        _unlabeled_ids (Tensor): Integer IDs for rejected samples.
        _ids_inferred (Tensor): IDs of inferred samples appended to the labeled pool.
    """
    _tensor_attr_after_init: List[str] = [
        "features_unlabeled", "features_labeled", "labels", 
        "_unlabeled_ids", "_ids_inferred", "_nan_val_ids_rejects", "_nan_val_labels"
    ]
    _lambdas_after_init : List[str] = ["_labels_nan_checker", "_ids_rejects_nan_checker"]
    _to_copy_with_separate_logic: List[str] = ["rng"]

    _class_members_containing_what_to_deepcopy: List[str] = ['_tensor_attr_after_init', '_lambdas_after_init', '_to_copy_with_separate_logic']

    _members_not_to_deepcopy: List[str] = ["retrieve_only_labeled"]

    def __init__(
        self,
        features_rejects: torch.Tensor,
        features_accepts: torch.Tensor,
        default_flag_accepts: torch.Tensor,
        ids_rejects : Optional[torch.Tensor] = None,
        retrieve_only_labeled: bool = True,
        seed: Optional[int] = None,
        safety_checks : bool = True,
        nan_value_labels : Optional[Union[float, int, torch.Tensor]] = None,
        nan_value_ids_rejects : Optional[Union[float, int, torch.Tensor]]  = None
    ):
        r"""
        Initialize a leakage-safe credit dataset sample.

        Args:
            features_rejects (Tensor):
                Feature matrix for rejected applications.
                Shape ``(..., N_unlabeled, F)``.
            features_accepts (Tensor):
                Feature matrix for accepted applications.
                Shape ``(..., N_labeled, F)``.
            default_flag_accepts (Tensor):
                Repayment outcomes for accepted applications.
                Shape ``(..., N_labeled)``.
            ids_rejects (Tensor, optional):
                Integer IDs for rejected samples. Must match
                ``features_rejects.shape[:-1]``.
                If ``None`` (default), IDs are assigned as a flattened ``arange``.
            retrieve_only_labeled (bool, optional):
                Whether ``__getitem__`` returns only labeled samples.
                Default: ``True``.
            seed (int, optional):
                Seed for initializing the internal RNG. Default: ``None``.
            safety_checks (bool, optional):
                Whether to validate shapes, dtypes, devices, and NaN structure.
                Default: ``True``.
            nan_value_labels (float, int, Tensor, optional):
                Value used to represent missing labels. If ``None`` (default),
                floating labels use ``nan`` and integer labels use ``-1``.
                If is a ``Tensor`` it must be a singleton.
            nan_value_ids_rejects (float, int, Tensor, optional):
                Value used to represent missing IDs. If ``None`` (default),
                floating IDs use ``nan`` and integer IDs use ``-1``.
                If is a ``Tensor`` it must be a singleton.

        Notes:
        - Input requirements (enforced when ``safety_checks=True``):
            - Leading dimensions must match:
            ``features_rejects.shape[:-2] == features_accepts.shape[:-2]`` and
            ``features_accepts.shape[:-1] == default_flag_accepts.shape``.
            - ``features_rejects`` and ``features_accepts`` must have identical
            ``dtype`` and be floating-point tensors and have the same final dimension.
            - All input tensors must reside on the same device.
            - ``features_rejects`` and ``features_accepts`` must be *compact*:
            it means that it exists at least a superbatch (each super batch has a 
            shape ``[N, F]``) for which all observations in the superbatch have at
            least one non-NaN feature, i. e. they all of the observations are valid.
            - ``default_flag_accepts`` and ``features_accepts`` must share the same
            NaN pattern: positions where labels are missing must have all-NaN
            features.
            - If ``ids_rejects`` is provided, it must match the shape
            ``features_rejects.shape[:-1]`` and use the same device.
        - Rejected samples never expose repayment outcomes.
        - The RNG controls all stochastic behavior (splits, permutations, etc.).
        """
        # ---------------------------------------------------------
        # 1. Safety checks on shapes, dtypes, devices
        # ---------------------------------------------------------
        if safety_checks:
            features_shapes_compatible = (features_rejects.shape[:-2] + features_rejects.shape[-1:]) == (features_accepts.shape[:-2] + features_accepts.shape[-1:])
            if not features_shapes_compatible:
                raise ValueError("Features must have same leading dimensions (.shape[:-2]) and final dimension .size(-1)")
            features_accepts_and_def_flags_have_compatible_shapes = features_accepts.shape[:-1] == default_flag_accepts.shape
            if not features_accepts_and_def_flags_have_compatible_shapes:
                raise ValueError("features_accepts.shape[:-1] == default_flag_accepts.shape must hold")
            if features_accepts.dtype != features_rejects.dtype:
                raise ValueError("features must have the same dtype")
            all_on_same_device = features_rejects.device==features_accepts.device==default_flag_accepts.device
            if not all_on_same_device:
                raise ValueError("features and flags must be on same device")
                
            if not all([torch.is_floating_point(f) for f in (features_accepts, features_rejects)]):
                raise ValueError("features rejects and accepts have to be floating points")
            
        # ---------------------------------------------------------
        # 2. Normalize labels and determine NaN encodings
        # ---------------------------------------------------------
        if default_flag_accepts.dtype == torch.bool:
            warn("default_flag_accepts will be changed to torch.int8, bool is not supported")
            default_flag_accepts = default_flag_accepts.to(torch.int8)

        if nan_value_labels is None:
            nan_value_labels = float('nan') if torch.is_floating_point(default_flag_accepts) else -1

        # ---------------------------------------------------------
        # 3. Initialize reject IDs and inferred-ID structure
        # ---------------------------------------------------------
        rej_batch_shape = features_rejects.shape[:-1]

        if ids_rejects is None:
            count_rej_obs = torch.prod(torch.tensor(rej_batch_shape))
            self._unlabeled_ids = torch.arange(count_rej_obs).reshape(*rej_batch_shape)
        else:
            if safety_checks:
                if ids_rejects.shape != rej_batch_shape:
                    raise ValueError("ids_rejects has the wrong shape. Should be features_rejects.shape[:-1]")
                if ids_rejects.device != features_rejects.device:
                    raise ValueError("ids_rejects is not on the same device as the other tensors")
                
            self._unlabeled_ids = ids_rejects

        if nan_value_ids_rejects is None:
            nan_value_ids_rejects = (
                float("nan") if torch.is_floating_point(self._unlabeled_ids) else -1
            )
        
        nan_value_ids_rejects_singleton = CreditDataSample._nan_value_to_singleton_tensor(
            nan_value_ids_rejects,
            dtype=self._unlabeled_ids.dtype,
            device=self._unlabeled_ids.device
        )
        self._ids_inferred = nan_value_ids_rejects_singleton.expand(default_flag_accepts.shape)
        
        # Apply NaN encodings
        self.labels = default_flag_accepts
        self.set_nan_val_labels(nan_value_labels)
        self.set_nan_val_ids_rejects(nan_value_ids_rejects)       
        
        # ---------------------------------------------------------
        # 4. Additional structural checks (compactness, NaN alignment)
        # ---------------------------------------------------------
        if safety_checks:
            # There has to be at least one not nan feature per (valid) observation
            # if an observation is valid it also has an _unlabeled_id
            mask_rej_feats_all_nan = features_rejects.isnan().all(dim=-1)
            mask_rej_feats_with_compatible_nans = ~mask_rej_feats_all_nan # No observations without any nans
            if ids_rejects is not None:
                mask_ids_rejects_is_nan = self._ids_rejects_nan_checker(self._unlabeled_ids)
                # Has to be XOR. mask_rej_feats_with_compatible_nans contains now False in every
                # place where an observation had all features as nan. It may not happen that where
                # not all observations have all nan features, the ids_rejects implies the observation
                # being unvalid. Opposite is True for the places where mask_ids_rejects_is_nan implies
                # an observation being valid
                mask_rej_feats_with_compatible_nans = mask_rej_feats_with_compatible_nans ^ mask_ids_rejects_is_nan

            if not mask_rej_feats_with_compatible_nans.all():
                raise ValueError((
                    "features_rejects had observations where all features were nan and ids_rejects does not imply it being a "
                    "position with null observations"
                ))
            
            at_least_one_batch_rej_has_complete_obs = (mask_rej_feats_all_nan.sum(dim=-1) == 0).any()
            if not at_least_one_batch_rej_has_complete_obs:
                raise ValueError("features_rejects is not compact!")
                
            # Accepts: NaN labels <-> NaN features
            mask_labels_is_nan = self._labels_nan_checker(self.labels)
            mask_acc_feats_all_nan = features_accepts.isnan().all(dim=-1)
            nan_in_labels_implies_nan_features = (mask_labels_is_nan == mask_acc_feats_all_nan).all()
            if not nan_in_labels_implies_nan_features:
                raise ValueError("NaN pattern mismatch between labels and features_accepts.")

            at_least_one_batch_acc_has_complete_obs = (mask_acc_feats_all_nan.sum(dim=-1)==0).any()
            if not at_least_one_batch_acc_has_complete_obs:
                raise ValueError("default_flags_accepts and features_accepts are not compact!")
        
        # ---------------------------------------------------------
        # 5. Final assignments + RNG
        # ---------------------------------------------------------
        self.features_unlabeled = features_rejects
        self.features_labeled = features_accepts

        self.retrieve_only_labeled = retrieve_only_labeled

        # RNG is device-specific, so we create it on the same device as the data
        self.rng = torch.Generator(device=features_rejects.device)
        if seed is not None:
            self.rng.manual_seed(int(seed))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device on which the dataset tensors reside."""
        return self.features_unlabeled.device

    @property
    def mask_inferred_lbls(self):
        """
        Boolean mask indicating which labeled samples originate from inferred
        rejected applications. Shape ``(..., N_labeled)``.
        """
        return ~self._ids_rejects_nan_checker(self._ids_inferred)

    @property
    def count_labeled(self):
        """Number of labeled samples per super-batch (``N_labeled``)."""
        return self.features_labeled.size(-2)
    
    @property
    def count_unlabeled(self):
        """Number of unlabeled samples per super-batch (``N_unlabeled``)."""
        return self.features_unlabeled.size(-2)
    
    @property
    def features_count(self):
        """Number of features per sample (``F``)."""
        return self.features_unlabeled.size(-1)

    # -------------------------------------------------------------------------
    # RNG utilities
    # -------------------------------------------------------------------------

    def manual_seed(self, seed):
        """
        Reseed the internal RNG deterministically.

        Args:
            seed (int): New seed value.

        Returns:
            ``CreditDataSample``: ``self``.
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
        r"""
        Move all dataset tensors and the RNG to a target device.

        Args:
            device (torch.device):
                Target device.
            seed (int, optional):
                Explicit seed for the new RNG. Overrides ``set_same_initial_seed``.
                Default: ``None``.
            set_same_initial_seed (bool, optional):
                If ``True`` and ``seed`` is ``None``, the new RNG is initialized
                with the previous RNG's initial seed. Default: ``True``.

        Returns:
            ``CreditDataSample``: ``self`` moved to ``device``.
        """
        
        for var in CreditDataSample._tensor_attr_after_init:
            setattr(self, var, getattr(self, var).to(device))

        # Preserve reproducibility across device transfers
        if seed is None and set_same_initial_seed:
            seed = self.rng.initial_seed()

        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

        return self

    # -------------------------------------------------------------------------
    # NaN-value utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_tensor_nan_checker(nan_value : Union[float, int, torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        r"""
        Build a function that checks whether elements of a tensor match the
        configured NaN value.

        Args:
            nan_value (float, int, Tensor):
                The value representing missingness. If torch.Tensor it must be a
                singleton (``nan_value.dim()==0`` and ``nan_value.numel()==1``)

        Returns:
            Callable[[Tensor], Tensor]:
                A function returning a boolean mask.
        """
        if isnan(nan_value):
            return lambda ten : ten.isnan()
        return lambda ten : ten == nan_value
    
    @staticmethod
    def _nan_value_to_singleton_tensor(
        nan_value : Union[float, int, torch.Tensor],
        dtype : Optional[torch.dtype] = None,
        device : Optional[torch.device] = None
    ) -> torch.Tensor:
        r"""
        Convert a scalar or tensor ``nan_value`` into a single-element tensor
        with the specified ``dtype`` and ``device``.

        Args:
            nan_value (float, int, Tensor):
                Value representing missingness.
            dtype (torch.dtype, optional): Desired dtype. Default: ``None``.
            device (torch.device, optional): Desired device. Default: ``None``.

        Returns:
            Tensor: A single-element tensor containing ``nan_value``.
        """
        if isinstance(nan_value, torch.Tensor):
            assert nan_value.numel() == 1, "nan_value has to have a single element"
            
            return nan_value.flatten().squeeze().to(dtype=dtype, device=device)
        
        return torch.tensor(nan_value, dtype=dtype, device=device)

    def set_nan_val_ids_rejects(self, nan_value : Union[float, int, torch.Tensor]) -> None:
        r"""
        Update the NaN encoding used for ``_unlabeled_ids`` and ``_ids_inferred``.

        All existing NaN positions (according to the previous checker) are rewritten
        using the new ``nan_value``.

        Args:
            nan_value (float, int, Tensor):
                New NaN encoding for reject IDs.
        """
        # Set to new nan value everywhere where it is value
        ids_rejects_nan_checker = getattr(
            self,
            "_ids_rejects_nan_checker",
            lambda t : torch.tensor(False, device=t.device).expand(t.shape)
        )
        nan_value_as_singleton = CreditDataSample._nan_value_to_singleton_tensor(
            nan_value,
            self._unlabeled_ids.dtype,
            self._unlabeled_ids.device
        )

        self._unlabeled_ids = torch.where(
            ids_rejects_nan_checker(self._unlabeled_ids), 
            nan_value_as_singleton, # will generate error if nan_value not castable to _unlabeled_ids.dtype
            self._unlabeled_ids
        )
        self._ids_inferred = torch.where(
            ids_rejects_nan_checker(self._ids_inferred), 
            nan_value_as_singleton, # will generate error if nan_value not castable to _unlabeled_ids.dtype
            self._ids_inferred
        )
        
        self._nan_val_ids_rejects = nan_value_as_singleton
        self._ids_rejects_nan_checker = CreditDataSample._build_tensor_nan_checker(nan_value)

    def set_nan_val_labels(self, nan_value : Union[float, int, torch.Tensor]):
        r"""
        Update the NaN encoding used for ``labels``.

        All existing NaN positions (according to the previous checker) are rewritten
        using the new ``nan_value``.

        Args:
            nan_value (float, int, Tensor):
                New NaN encoding for labels.
        """
        # Set to new nan value everywhere where it is value
        labels_nan_checker = getattr(
            self,
            "_labels_nan_checker",
            lambda t : torch.tensor(False, device=t.device).expand(t.shape)
        )
        nan_value_as_singleton = CreditDataSample._nan_value_to_singleton_tensor(
            nan_value,
            self.labels.dtype,
            self.labels.device
        )

        self.labels = torch.where(
            labels_nan_checker(self.labels), 
            nan_value_as_singleton, # will generate error if nan_value not castable to _unlabeled_ids.dtype
            self.labels
        )

        self._nan_val_labels = nan_value_as_singleton
        self._labels_nan_checker = CreditDataSample._build_tensor_nan_checker(nan_value)

    # -------------------------------------------------------------------------
    # Clone class related
    # -------------------------------------------------------------------------

    @staticmethod
    def new_instance_with_full_info(
            features_unlabeled: torch.Tensor,
            unlabeled_ids: torch.Tensor,
            features_labeled: torch.Tensor,
            labels: torch.Tensor,
            ids_inferred: torch.Tensor,
            nan_val_ids_rejects: torch.Tensor,
            nan_val_labels: torch.Tensor,
            retrieve_only_labeled: bool = True,
            seed: Optional[int] = None,
            rng_state: Optional[torch.Tensor] = None,
            safety_checks: bool = True
    ) -> "CreditDataSample":
        r"""
        Construct a new :class:`CreditDataSample` instance using fully specified
        tensors and optional RNG state.

        This method bypasses the usual data ingestion logic and directly injects
        all internal tensors required after initialization. It is primarily used
        for cloning and reconstruction, where the caller already holds validated
        tensors in the correct shapes.

        Args:
            features_unlabeled (Tensor):
                Feature matrix for the unlabeled (reject) population.
            unlabeled_ids (Tensor):
                Identifier tensor for the unlabeled population.
            features_labeled (Tensor):
                Feature matrix for the labeled (accept) population.
            labels (Tensor):
                Label tensor for the labeled population.
            ids_inferred (Tensor):
                Tensor containing inferred IDs. Must satisfy
                ``ids_inferred.shape == labels.shape``.
            nan_val_ids_rejects (Tensor):
                Scalar tensor representing the NaN value for reject IDs.
            nan_val_labels (Tensor):
                Scalar tensor representing the NaN value for labels.
            retrieve_only_labeled (bool, optional):
                Whether the instance should expose only labeled samples when
                retrieving data. Default: ``True``.
            seed (int, optional):
                Seed used to initialize the internal :class:`torch.Generator`.
            rng_state (Tensor, optional):
                Serialized RNG state to restore via ``Generator.set_state``.
                Defaults to ``None``
            safety_checks (bool, optional):
                If ``True``, perform consistency checks on NaN patterns and shapes.
                Defaults to ``True``

        Returns:
            CreditDataSample:
                A fully constructed instance with all internal members populated.
        """
        new_instance = CreditDataSample(
            features_rejects=features_unlabeled,
            features_accepts=features_labeled,
            default_flag_accepts=labels,
            ids_rejects=unlabeled_ids,
            retrieve_only_labeled=retrieve_only_labeled,
            seed=seed,
            safety_checks=safety_checks,
            nan_value_labels=nan_val_labels,
            nan_value_ids_rejects=nan_val_ids_rejects
        )
        if safety_checks:
            if ids_inferred.shape != new_instance.labels.shape:
                raise ValueError("ids_inferred should have the same shape as labels (after init)")
            mask_id_rej_is_nan = new_instance._ids_rejects_nan_checker(ids_inferred)
            mask_labels_is_nan = new_instance._labels_nan_checker(new_instance.labels)

            any_label_is_nan_and_id_is_not = torch.any(mask_id_rej_is_nan < mask_labels_is_nan)
            if any_label_is_nan_and_id_is_not:
                raise ValueError("NaNs pattern in ids_inferred not compatible with labels NaN pattern")
            
        new_instance._ids_inferred = ids_inferred

        if rng_state is not None:
            new_instance.rng.set_state(rng_state)

        return new_instance
    
    def is_valid_clone(self, clone : "CreditDataSample") -> Tuple[bool, str]:
        r"""
        Validate that ``clone`` is a correct structural and semantic clone of ``self``.

        The method checks:
        - type equality of corresponding attributes,
        - absence of shared references for members that must be deep-copied,
        - equality of tensor data (including matching ``NaN`` patterns),
        - equivalence of lambda-based NaN checkers,
        - equality of RNG state for the internal :class:`torch.Generator`,
        - correct handling of members that must not be deep-copied.

        Attributes are grouped according to class-level lists such as
        ``_tensor_attr_after_init`` and ``_lambdas_after_init`` to determine the
        expected cloning semantics.

        Args:
            clone (CreditDataSample):
                The candidate clone to validate.

        Returns:
            (bool, str):
                ``(True, "")`` if the clone is valid. Otherwise ``(False, msg)``,
                where ``msg`` describes the first detected inconsistency.
        """
        types_match = lambda obj1, obj2 : type(obj1) is type(obj2)
        tensors_have_same_data = lambda ten1, ten2 : ten1.shape == ten2.shape and torch.all((ten1 == ten2) | (ten1.isnan() & ten2.isnan()))

        if not types_match(self, clone):
            return False, "self and clone type mismatch"
        
        cls = self.__class__
        
        
        for container_name in cls._class_members_containing_what_to_deepcopy:
            names_to_be_copied = getattr(cls, container_name)

            warn_for_not_knowing_how_to_check_equal_data = False
            for obj_name in names_to_be_copied:
                orig_obj = getattr(self, obj_name)
                cloned_obj = getattr(clone, obj_name)

                if not types_match(orig_obj, cloned_obj):
                    return False, f"Member {obj_name} did not have the same type in clone and in self"

                objs_share_reference = orig_obj is cloned_obj
                if objs_share_reference:
                    return False, f"clone.{obj_name} is a reference to self.{obj_name}"

                if container_name == '_tensor_attr_after_init':
                    if not tensors_have_same_data(orig_obj, cloned_obj):
                        return False, f"Copying {obj_name} did not retrieve tensors with the same data"
                elif container_name == '_lambdas_after_init':
                    # minimal lambda equivalence check 
                    if orig_obj.__name__ != "<lambda>" or cloned_obj.__name__ != "<lambda>": 
                        return False, f"{obj_name}: expected lambda but got non-lambda" 
                    if orig_obj.__code__.co_freevars != cloned_obj.__code__.co_freevars:
                        return False, f"{obj_name}: lambda closure mismatch" 
                    
                elif container_name == '_to_copy_with_separate_logic':
                    if obj_name == 'rng':
                        if self.rng.get_state().tolist() != clone.rng.get_state().tolist():
                            return False, "States of self.rng and dummy_clone.rng are not the same"
                    else:
                        warn_for_not_knowing_how_to_check_equal_data = True
                else:
                    warn(f"{container_name} not recognized - no logic or white listing for equal data possible")
                    warn_for_not_knowing_how_to_check_equal_data = True

            if warn_for_not_knowing_how_to_check_equal_data:
                warn(f"The attributes in {container_name} = [{','.join(names_to_be_copied)}] did not share adress but cannot be checked on equality of data")

        for obj_name in cls._members_not_to_deepcopy:
            orig_obj = getattr(self, obj_name)
            cloned_obj = getattr(clone, obj_name)
            if not types_match(orig_obj, cloned_obj):
                return False, f"Member {obj_name} did not have the same type in clone and in self"
            
            if orig_obj is cloned_obj:
                continue
                
            if isinstance(orig_obj, torch.Tensor) and not tensors_have_same_data(orig_obj, cloned_obj): 
                return False, f"{obj_name}: tensor mismatch"

        checked = ( 
            cls._tensor_attr_after_init 
            + cls._lambdas_after_init 
            + cls._to_copy_with_separate_logic 
            + cls._members_not_to_deepcopy
        )
        basic_types_comparable_with_equality = (int, float, dict, list, set, frozenset)
        members_not_checked = set(self.__dict__.keys()) - set(checked)
        
        for obj_name in members_not_checked:
            orig_obj = getattr(self, obj_name)
            cloned_obj = getattr(clone, obj_name)
            if not types_match(orig_obj, cloned_obj):
                return False, f"Member {obj_name} did not have the same type in clone and in self"
            
            if isinstance(orig_obj, basic_types_comparable_with_equality) and orig_obj != cloned_obj:
                return False, f"Member {obj_name} did not contain the same data in clone and in self"
            
            warn(f"Do not know how to check if cloning was ok with member {obj_name}")

        return True, ""
    
    def clone(self, safety_data_integrety_tests: bool = False):
        r"""
        Create a new :class:`CreditDataSample` instance that is a structural and
        semantic clone of ``self``.

        The cloning procedure:
        - clones all tensor attributes listed in ``_tensor_attr_after_init``,
        - reconstructs the instance via :meth:`new_instance_with_full_info`,
        - restores the internal RNG state,
        - optionally performs integrity checks on tensor members before cloning.

        This method does **not** perform a shallow copy of the object. Instead,
        it reconstructs a fresh instance with newly cloned tensors and identical
        configuration values.

        Args:
            safety_data_integrety_tests (bool, optional):
                If ``True``, verify that all tensor attributes expected after
                initialization exist and are valid ``torch.Tensor`` objects.

        Returns:
            CreditDataSample:
                A clone of the current instance with no shared mutable state.
        """
        if safety_data_integrety_tests:
            for ten_name in CreditDataSample._tensor_attr_after_init:
                if not isinstance(getattr(self, ten_name, None), torch.Tensor):
                    raise ValueError(f"Member {ten_name} was non existent or not a torch.Tensor")
                
        kwargs_new_instance_with_full_info = {
            (k[1:] if k[0]=='_' else k) : getattr(self, k).clone() for k in CreditDataSample._tensor_attr_after_init
        }
        
        return CreditDataSample.new_instance_with_full_info(
            retrieve_only_labeled = self.retrieve_only_labeled,
            rng_state = self.rng.get_state(),
            safety_checks=safety_data_integrety_tests,
            **kwargs_new_instance_with_full_info
        )


    # -------------------------------------------------------------------------
    # (Reject) Inference related
    # -------------------------------------------------------------------------

    def _generate_random_train_test_idxs(
        self,
        shape_up_to_N_dim : Union[tuple[int], torch.Size],
        test_proportion: float,
    ) -> torch.Tensor:
        r"""
        Generate random train/test index splits along the sample dimension.

        Args:
            shape_up_to_N_dim (tuple or torch.Size):
                Super-batch shape ending with ``N`` (number of samples).
            test_proportion (float):
                Fraction of samples to assign to the test set.

        Returns:
            (Tensor, Tensor):
                ``(train_indices, test_indices)``, each suitable for ``gather``.
    """
        test_count = round(test_proportion * shape_up_to_N_dim[-1])
        scores = torch.randn(shape_up_to_N_dim, generator=self.rng, device=self.device).argsort(dim=-1)
        idx_N_dim_gather_test, idx_N_dim_gather_train = scores[..., :test_count], scores[..., test_count:]

        return idx_N_dim_gather_train, idx_N_dim_gather_test

    def train_test_split(self, test_proportion: float, check_data_integrity_before_returning : bool = False):
        """
        Split the dataset into train and test subsets without leakage.

        The split is performed independently for labeled and unlabeled pools,
        preserving super-batch structure and using gather-based indexing.

        Args:
            test_proportion (float):
                Fraction of samples to assign to the test set.
            check_data_integrity_before_returning (bool):
                Whether to check if all data in the class fullfills the expected
                characteristics. Only necessary/sensible if the tensors in the class
                have been changed manually, like adding more data - which is not the
                intentede purpose of the clase. Defaults to ``False``.

        Returns:
            (CreditDataSample, CreditDataSample):
                (``train_sample``, ``test_sample``), each containing consistent subsets of
                features, labels, IDs, and inferred-ID tracking.
        """
        if not (0.0 <= test_proportion <= 1.0):
            raise ValueError("test_proportion must be between 0 and 1")

        # Generate masks
        gather_idx_train_unlbld, gather_idx_test_unlbld = self._generate_random_train_test_idxs(
            self.features_unlabeled.shape[:-1], test_proportion
        )
        gather_idx_train_lbld, gather_idx_test_lbld = self._generate_random_train_test_idxs(
            self.features_labeled.shape[:-1], test_proportion
        )

        gather_features = lambda gather_from, idx_gather : gather_from.gather(
            dim=-2,
            index=idx_gather.unsqueeze(-1).expand(*idx_gather.shape, self.features_count)
        )

        shared_args_for_new_instances = lambda _ : {
            "retrieve_only_labeled" : self.retrieve_only_labeled,
            "nan_val_ids_rejects" : self._nan_val_ids_rejects.clone(),
            "nan_val_labels" : self._nan_val_labels.clone(),
            "rng_state" : self.rng.get_state()
        }

        # Slice data
        train_sample = CreditDataSample.new_instance_with_full_info(
            features_unlabeled =    gather_features(self.features_unlabeled, gather_idx_train_unlbld),
            unlabeled_ids =         self._unlabeled_ids.gather(dim=-1, index=gather_idx_train_unlbld),
            features_labeled =      gather_features(self.features_labeled, gather_idx_train_lbld),
            labels =                gather_features(self.features_labeled, gather_idx_train_lbld),
            ids_inferred =          self._ids_inferred.gather(dim=-1, index=gather_idx_train_lbld),
            safety_checks =         check_data_integrity_before_returning
            **shared_args_for_new_instances()
        )

        test_sample = CreditDataSample.new_instance_with_full_info(
            features_unlabeled =    gather_features(self.features_unlabeled, gather_idx_test_unlbld),
            unlabeled_ids =         self._unlabeled_ids.gather(dim=-1, index=gather_idx_test_unlbld),
            features_labeled =      gather_features(self.features_labeled, gather_idx_test_lbld),
            labels =                gather_features(self.features_labeled, gather_idx_test_lbld),
            ids_inferred =          self._ids_inferred.gather(dim=-1, index=gather_idx_test_lbld),
            safety_checks =         check_data_integrity_before_returning
            **shared_args_for_new_instances()
        )

        return train_sample, test_sample
    
    def label_rejects(
            self, 
            inferred_labels : torch.Tensor, 
            mask_inferred_rej_lbls : torch.Tensor,
            inplace : bool = False,
            safety_checks : bool = True
    ) -> Union[None, "CreditDataSample"]:
        r"""
        Append inferred labels from the unlabeled pool into the labeled pool and
        compact the remaining unlabeled pool.

        This operation:
            - inserts inferred labels into available NaN slots in ``labels``
            - pads the labeled pool if more slots are required with corresponding
              saved NaN value
            - appends corresponding features and IDs
            - removes inferred samples from the unlabeled pool
            - preserves super-batch structure

        Args:
            inferred_labels (Tensor):
                Inferred labels for rejected samples. Shape ``(..., N_unlabeled)``.
            mask_inferred_rej_lbls (Tensor):
                Boolean mask selecting which unlabeled samples receive inferred labels.
                Same shape as ``inferred_labels``.
            inplace (bool, optional):
                If ``True``, modify the dataset in place. Default: ``False``.
            safety_checks (bool, optional):
                Whether to validate shapes, devices, and label consistency.
                Default: ``True``.

        Returns:
            CreditDataSample or None:
                New dataset instance if ``inplace=False``, otherwise ``None``.
        """
        # Early scape if no inferred rej lables
        if not mask_inferred_rej_lbls.any():
            if inplace:
                return
            return self.clone()
            
        if safety_checks:
            if mask_inferred_rej_lbls.dtype != torch.bool:
                raise ValueError("mask_infered_rej_lbls should be of type bool")
            if not (mask_inferred_rej_lbls.shape == inferred_labels.shape == self.features_unlabeled.shape[:-1]):
                raise ValueError("mask_inferred_rej_lbls and inferred_labels should have the same shape as self.features_unlabeled.shape[:-1]")
            
            inferred_labels_with_vals_as_saved_labels = torch.isin(inferred_labels[mask_inferred_rej_lbls].unique(), self.labels).all()
            if not inferred_labels_with_vals_as_saved_labels:
                raise ValueError("Tensor inferred_labels[mask_inferred_rej_lbls] should contain only values like in self.labels")
            
            input_tensors_on_same_device_as_self = inferred_labels.device == mask_inferred_rej_lbls.device == self.device
            if not input_tensors_on_same_device_as_self:
                raise ValueError("inferred_labels and mask_inferred_rej_lbls must be on same device as self")
            
        
        *batch_shape, N_labels = self.labels.shape
        B = torch.tensor(batch_shape).prod()

        current_labels = self.labels.reshape(B, N_labels)
        mask_nans_labels = self._labels_nan_checker(current_labels)

        N_unlabeled = mask_inferred_rej_lbls.size(-1)
        mask_inf = mask_inferred_rej_lbls.reshape(B, N_unlabeled)
        inf_lbls = inferred_labels.reshape(B, N_unlabeled)

        slots_available = mask_nans_labels.sum(dim=-1)
        slots_needed = mask_inf.sum(dim=-1)

        needed_padding = torch.maximum(slots_needed - slots_available, torch.tensor(0, device=self.device))
        max_needed_padding = needed_padding.max()

        
        mask_valid_lbls = ~mask_nans_labels
        batch_idx_valid_lbls, N_idx_valid_lbls = _mask2d_to_int_idxs(mask_valid_lbls)
        batch_idx_inf_lbls, N_idx_inf_lbls = _mask2d_to_int_idxs(mask_inf, N_labels - slots_available.unsqueeze(-1))

        def _append_obs(append_to : torch.Tensor, to_append : torch.Tensor, pad_spec : tuple[int], pad_val):
            appended = torch.nn.functional.pad(
                append_to, 
                pad=pad_spec,
                mode='constant',
                value=pad_val
            )
            appended[batch_idx_valid_lbls, N_idx_valid_lbls] = append_to[mask_valid_lbls].to(appended.dtype)
            appended[batch_idx_inf_lbls, N_idx_inf_lbls] = to_append[mask_inf].to(appended.dtype)
            return appended

        new_labels = _append_obs(
            append_to=current_labels, 
            to_append=inf_lbls, 
            pad_spec=(0, max_needed_padding),
            pad_val = self._nan_val_labels
        )

        current_unlbld_ids = self._unlabeled_ids.reshape(B, N_unlabeled)

        new_ids_inferred = _append_obs(
            append_to=self._ids_inferred.reshape(B, N_labels), 
            to_append=current_unlbld_ids,
            pad_spec=(0, max_needed_padding),
            pad_val = self._nan_val_ids_rejects
        )

        current_feats_unlbld = self.features_unlabeled.reshape(B, N_unlabeled, -1)
        new_features_labeled = _append_obs(
            append_to=self.features_labeled.reshape(B, N_labels, -1), 
            to_append=current_feats_unlbld,
            pad_spec=(0, 0, 0, max_needed_padding),
            pad_val=torch.nan
        )
        # Keep only the observations that were rejected and also were not nan
        mask_non_inferred_to_keep = ~self._ids_rejects_nan_checker(current_unlbld_ids) & ~mask_inf
        new_N_unlabeled = mask_non_inferred_to_keep.sum(dim=-1).max()
        batch_idx_resized_unlbld, N_idx_resized_unlbld = _mask2d_to_int_idxs(mask_non_inferred_to_keep)

        def _resize_obs(to_resize : torch.Tensor, nan_value):
            resized = to_resize.new_full(torch.Size([B, new_N_unlabeled]) + to_resize.shape[2:], nan_value)
            resized[batch_idx_resized_unlbld, N_idx_resized_unlbld] = to_resize[mask_non_inferred_to_keep]
            return resized

        features_unlabeled = _resize_obs(
            current_feats_unlbld, 
            nan_value=torch.nan
        )
        unlabeled_ids = _resize_obs(
            current_unlbld_ids, 
            nan_value=self._nan_val_ids_rejects
        )

        new_labels, new_ids_inferred, new_features_labeled, features_unlabeled, unlabeled_ids = [
            t.reshape(torch.Size(batch_shape) + t.shape[1:]) for t in 
            [new_labels, new_ids_inferred, new_features_labeled, features_unlabeled, unlabeled_ids]
        ]
            
        if inplace:
            self.features_labeled = new_features_labeled
            self.labels = new_labels
            self._ids_inferred = new_ids_inferred

            self.features_unlabeled = features_unlabeled
            self._unlabeled_ids = unlabeled_ids
            return
        
        return CreditDataSample.new_instance_with_full_info(
            features_unlabeled=features_unlabeled,
            unlabeled_ids=unlabeled_ids,
            features_labeled=new_features_labeled,
            labels=new_labels,
            ids_inferred=new_ids_inferred,
            nan_val_ids_rejects=self._nan_val_ids_rejects.clone(),
            nan_val_labels=self._nan_val_labels.clone(),
            retrieve_only_labeled=self.retrieve_only_labeled,
            rng_state=self.rng.get_state(),
            safety_checks=safety_checks
        )

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Number of samples returned by __getitem__, depending on retrieval mode.

        Returns:
            int: ``N_labeled`` if ``retrieve_only_labeled=True``, else ``N_unlabeled``.
        """
        return self.count_labeled if self.retrieve_only_labeled else self.count_unlabeled

    def __getitem__(self, idx: int) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Retrieve a single sample from either the labeled or unlabeled pool.

        Returns:
            dict:
                If retrieving labeled samples:
                    {
                        "features": Tensor (..., F),
                        "default_flag": Tensor (...,),
                        "is_inferred": Tensor (...,),
                        "accepted": True
                    }

                If retrieving unlabeled samples:
                    {
                        "features": Tensor (..., F),
                        "default_flag": None,
                        "is_inferred" : None,
                        "accepted": False
                    }
        """
        if self.retrieve_only_labeled:
            return {
                "features": self.features_labeled[..., idx, :],
                "default_flag": self.labels[..., idx],
                "is_inferred" : self.mask_inferred_lbls[..., idx],
                "accepted": True,
            }

        return {
            "features": self.features_unlabeled[..., idx, :],
            "default_flag": None,
            "is_inferred" : None,
            "accepted": False,
        }




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
            retrieve_only_labeled=retrieve_only_accepted,
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

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def data_stats(self) -> Dict[str, Union[float, int]]:
        """Compute descriptive statistics of the current dataset.

        The statistics are based on the acceptance flags and default outcomes and
        are computed across all samples currently stored in the dataset. In the
        simulation setting, default outcomes for rejected applications are also
        observed and are used here purely for diagnostic purposes. In real-world
        reject inference, such labels would typically be unobserved.

        Returns:
            Dict[str, Union[float, int]]:
                Dictionary containing:
                    - ``sample_size``: Total number of observations.
                    - ``accept_ratio``: Fraction of accepted applications among all observations.
                    - ``bad_ratio_accepts``: Default rate among accepted applications.
                    - ``bad_ratio_rejects``: Default rate among rejected applications
                      (simulation-only quantity).
                    - ``bad_ratio_unbiased``: Overall default rate across all observations.

        Raises:
            ZeroDivisionError:
                If there are no accepted and no rejected applications when computing
                the corresponding unbiased bad rate.
        """
        bads_count_among_accepts = self.default_flag[self.accepted_idx].sum().item()
        bads_count_among_rejects = self.default_flag[self.reject_idx].sum().item()
        all_obs_count = self.count_all

        stats = {
            "sample_size": all_obs_count,
            "accept_ratio": self.count_accepts / all_obs_count,
            "bad_ratio_accepts": float("nan")
            if self.count_accepts == 0
            else bads_count_among_accepts / self.count_accepts,
            "bad_ratio_rejects": float("nan")
            if self.count_rejects == 0
            else bads_count_among_rejects / self.count_rejects,
            "bad_ratio_unbiased": (bads_count_among_accepts + bads_count_among_rejects)
            / all_obs_count,
        }

        return stats




if __name__ == "__main__":
    print('*' * 10,"Initializing credit data simulation", '*' * 10)

