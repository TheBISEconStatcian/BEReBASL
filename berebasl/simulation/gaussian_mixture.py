from math import log, sqrt
import torch
from warnings import warn

from typing import Dict, Optional, Tuple, Union


def random_vcov_matrix(
        f: int,
        generator: Optional[torch.Generator] = None,
        var_range: Tuple[float, float] = (0.0, 1.0),
        prefer_normal_base_sampling: bool = True,
        device: torch.device = torch.get_default_device(),
        dtype: torch.dtype = torch.get_default_dtype(),
        eps: float = 1e-6
    ) -> torch.Tensor:
    """Generate a random positive definite covariance matrix.

    The covariance matrix is produced by:

    1. Generating a base sampling of a matrix :math:`A` (normal or uniform).
    2. Creating a correlation matrix via cosine similarity between the row vectors
       of the base sampling:

       .. math::

          C = (c_{i,j}) = \\left( \\frac{\\langle A_i, A_j \\rangle}{\\|A_i\\| \\|A_j\\|} \\right)

    3. Sampling a variance vector uniformly in ``var_range``, which is used to
       rescale the correlation matrix.
    4. Ensuring positive definiteness via addition of a small diagonal
       perturbation ``eps``.
    5. Make sure exact symmetry, so that rounding point instability does not
       lead to unsymmetric results. 

    Args:
        f (int): Dimension of the covariance matrix.
        generator (torch.Generator, optional): Random number generator for reproducibility.
        var_range (Tuple[float, float], optional): Range for diagonal variances. Defaults to (0.0, 1.0).
        prefer_normal_base_sampling (bool, optional): If True, use normal distribution for base sampling.
            If False, use uniform distribution. Defaults to True.
        device (torch.device, optional): Device on which to allocate the tensor.
            Defaults to ``torch.get_default_device()``.
        dtype (torch.dtype, optional): Data type of the returned tensor.
            Defaults to ``torch.get_default_dtype()``.
        eps (float, optional): Small positive value added to the diagonal to ensure positive definiteness.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: A symmetric, positive definite covariance matrix of shape ``(f, f)``.

    Raises:
        ValueError: If ``var_range`` is not a valid (min, max) tuple.

    Example:
        >>> g = torch.Generator().manual_seed(42)
        >>> cov = random_vcov_matrix(4, generator=g)
        >>> cov.shape
        torch.Size([4, 4])
    """

    # Step 1: Generate base sampling
    if prefer_normal_base_sampling:
        A = torch.randn((f, f), generator=generator, dtype=dtype, device=device) # random normal matrix, sparser correlations for high k
    else:
        A = 2*torch.rand((f,f), generator=generator, dtype = dtype, device=device) - 1 # random uniform matrix, correlations closer to 0, the higher k

    # Step 2: Define correlation matrix from base sampling
    Q = A @ A.T # Make sure of symmetry while using full randomness
    D = torch.sqrt(torch.diag(Q)) # Help vector for normalization
    corr_mat = Q / torch.outer(D, D) # corr_mat[i, j] = cosine_similarity(A[i], A[j]), so range [-1, 1] guaranteed

    # Step 3: Rescale corr_mat with sampled variances
    ## Variance sampling from uniform distribution
    variances = torch.rand(f, generator=generator, dtype = dtype, device=device) * (var_range[1] - var_range[0]) + var_range[0]
    ## Rescaling via outer prouct of standard deviations
    stds = torch.sqrt(variances)
    norm_factors_pearson_corr = torch.outer(stds, stds) # guaranteed to be symmetric, denominators of pearson correlation
    vcov = corr_mat * norm_factors_pearson_corr

    # Step 4: Avoid semi positive definitness of the matrix
    vcov = vcov + eps * torch.eye(f, device=device, dtype=dtype)

    # Step 5: Ensure **exact** symmetry without compromising randomness
    i, j = torch.tril_indices(f, f, offset=-1)
    vcov[i, j] = vcov[j, i]

    
    return vcov

def eigen_decomp_proj_to_pd(
    mat: torch.Tensor,
    eps: float = 1e-6,
    ensure_symmetry: bool = False
) -> torch.Tensor:
    """Project a matrix onto the positive definite (PD) cone via eigen-decomposition.

    The procedure ensures the output is symmetric and positive semidefinite by:
    
    1. Optionally symmetrizing the input matrix.
    2. Performing eigen-decomposition.
    3. Clipping eigenvalues below ``eps`` to enforce non-negativity.
    4. Reconstructing the matrix from clipped eigenvalues and eigenvectors.
    5. Symmetrizing the result again to avoid numerical drift.

    Args:
        mat (torch.Tensor): Input square matrix of shape ``(f, f)``.
        eps (float, optional): Minimum eigenvalue threshold to enforce positive definiteness.
            Defaults to ``1e-6``.
        ensure_symmetry (bool, optional): If True, symmetrize the input before decomposition.
            Defaults to False.

    Returns:
        torch.Tensor: Symmetric positive semidefinite matrix of shape ``(f, f)``.

    Example:
        >>> M = torch.tensor([[1.0, 2.0], [2.0, -3.0]])
        >>> M_psd = eigen_decomp_proj_to_pd(M)
        >>> torch.linalg.eigvalsh(M_psd)
        tensor([1.0133e-06, 1.8284e+00])
    """
    # Ensure symmetry
    if ensure_symmetry:
        mat = (mat + mat.T) / 2
    
    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(mat)
    
    # Clip eigenvalues to non-negative
    eigvals_clipped = torch.clamp(eigvals, min=eps)
    
    # Reconstruct
    mat_psd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T
    
    # Ensure symmetry again
    return (mat_psd + mat_psd.T) / 2

def mvn_random_sample(
        mean : torch.Tensor, 
        cov_chol_decomp : Optional[torch.Tensor],
        n : int, 
        rng : Optional[torch.Generator] = None, 
        args_checks : bool = True
    ):
    """Generate n-vectors sampled of a multivariate normal (MVN) distribution with parameters
    mean and cov. Based on the implementation of (r)sample from 
    torch.distributions.MultivariateNormal according to torch version 2.9.1. It uses
    cholesky-decomposition method.

    Args:
        mean (torch.Tensor): Location parameter of a MVN. Shape ``(f,)`` or ``(b, f)`` or ``(1,5)``.
        cov_chol_decomp (torch.Tensor): Variance-Covariance matrix of MVN after cholesky decomposition. 
            Shape ``(f, f)``` or ``(b, f, f)`` or ``(1, f, f)``, ``cov.dim()==mean.dim()+1`` should hold.
        n (int): Count of vectors to be sampled (per batch).
        rng (Optional[torch.Generator]): If passed, sampling is done using this
            generator.
        args_checks (bool): If true, it will be checked whether the shapes of mean and
            cov are as expected, whether symmetry (w. r. t. to the last two dims for each batch)
            is given within the range of ``symmetry_rtol_atol`` for ``cov`` and type checks
            are done for ``n`` and ``rng``.`
        symmetry_rtol_atol (Tuple[float,float]): Corresponds to the (rtol, a_tol) parameters
            of ``torch.allclose``, passed as ``*args``, so ordering is important. Ignored if
            ``not args_checks``.
    Returns:
        torch.Tensor:
            A tensor of shape ``(n, k)`` or ``(n, b, f)`` containing the ``n`` sampled vectors (for each batch).

    Example:
        >>> count_covariates = 5
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> torch.set_default_dtype(torch.float32)
        >>> rng = torch.Generator(device)
        >>> mu = torch.zeros(count_covariates)
        >>> sigma_bad, _ = generate_sigma_bad_and_good(k = count_covariates, proportion_var_dif=1.0, generator = rng, device=device, dtype= torch.get_default_dtype())
        >>> sample = mvn_random_sample(mean=mu, cov=toch.linalg.cholesky(sigma_bad), n=100, rng=rng)
        >>> sample.shape
        torch.Size([100, 5])
    """
    if args_checks:
        #shape checks
        assert (mean.dim() in [1, 2, 3]) and (cov_chol_decomp.dim()==mean.dim()+1), "mean must be a single vector (rank 1 tensor) and cov a matrix (rank 2 tensor)"
        assert (mean.size(-1) == cov_chol_decomp.size(-1)) and (cov_chol_decomp.size(-1) == cov_chol_decomp.size(-2)), "mean must have shape [k] and cov shape [f, f]"
        #ensure n is an int
        n = int(n)
        assert isinstance(rng, torch.Generator) or rng is None, "rng needs to be None or a rng"

    shape = torch.Size([n]) + mean.shape
    
    eps = torch.empty(shape, dtype=mean.dtype, device = mean.device).normal_(generator=rng)

    deviations = torch.matmul(cov_chol_decomp, eps.unsqueeze(-1)).squeeze(-1) # apply decomp to each sampled vector


    return mean + deviations


class GaussianMixture:
    r"""
    Gaussian mixture distribution with fixed means, covariances and optional mixture weights.

    This class supports:
      - unbatched mixtures
      - batched mixtures (different weights per batch)
      - deterministic or stochastic sampling of mixture assignments
      - sampling from independent multivariate Gaussians if no weights are provided

    When ``weights`` is provided, sampling follows a Categorical distribution over
    components unless ``deterministic_weights=True`` is used. In that case the number
    of samples drawn from each component is a rounded version of ``n * weights``.

    Args:
        mean (Tensor): Mean tensor of shape ``(f,)``, ``(k, f)``, ``(b, f)`` or ``(b, k, f)``.
            Supports both batched and unbatched mixtures.
        cov (Tensor): Covariance tensor of shape ``(f, f)``, ``(m, f, f)``, ``(b, f, f)`` or ``(b, m, f, f)``.
            Must be symmetric. Its last two dims must match the dimensionality of the means.
        weights (Tensor, optional): Mixture weights of shape ``(m,)`` or ``(b, m)``.
            Must sum to 1 along the last dimension. If not provided, the class represents
            independent Gaussian sampling instead of a true mixture.
        seed (int, optional): Seed for internal RNG used for sampling and deterministic
            assignment correction.
        cov_symmetry_rtol_atol (Tuple[float], optional): Relative/absolute tolerance for
            covariance symmetry assertions.
        check_params (bool): If ``True``, validate shapes, symmetry and weight normalization.

    Shape:
        - ``b``: batch size of independent mixtures (if weights are batched).
        - ``m``: number of components in the mixture.
        - ``k``: dimensionality of the multivariate normal.

    Note:
        - For dimensions ``b`` and ``m``, broadcasting rules apply - so that for that dimension in each
          parameter either ``1`` or the corresponding size can be passed. ``weights`` must however have size
          ``m`` in its last dimension. This is a way to allow for flexible specification of mixtures that 
          are easily readible, e.g. a single set of weights can be shared across batches, or a single Gaussian
          can be used across mixture components. During initalization the shapes are normalized according to the
          case implied by the arguments dimensions, i.e:
            - single gaussian: ``mean.shape=(f,)``
            - independent gaussian: ``mean.shape=(b, f)``,
            - mixture: ``mean.shape=(k, f)`` or 
            - independent mixtures: ``mean.shape=(b, k, f)``.
          whereby ``b`` and ``m`` are inferred from the dimensions of the provided arguments.
        - If ``weights`` is not provided and ``mean`` has shape ``(b, f)``, the class represents
          independent Gaussian sampling with the provided means and covariances, and the sampling 
          methods will draw from each Gaussian independently.
        - The ``sample`` method returns samples of shape ``(n, k)`` for unbatched cases and
          ``(b, n, k)`` for batched cases, where ``n`` is the number of samples drawn per batch.

    Example::

        gm = GaussianMixture(mean, cov, weights)
        samples = gm.sample(100)                      # stochastic mixture
        det_samples = gm.sample(100, deterministic_weights=True)

    """
    def __init__(
            self,
            mean : torch.Tensor,
            cov : torch.Tensor,
            weights : Optional[torch.Tensor] = None,
            seed : Optional[int] = None,
            cov_symmetry_rtol_atol : Tuple[float] = [0.0, 0.0],
            check_params : bool = True
        ):
        if check_params:
            GaussianMixture.dist_params_check(mean, cov, weights, cov_symmetry_rtol_atol)
        
        ## From here assume all variables being as if they pass dist_params_check
        self.mean = mean.clone()
        F = mean.size(-1)
        self.cov_chol_decomp: torch.Tensor = torch.linalg.cholesky(cov)
        is_mixture = weights is not None

        if is_mixture:
            is_batched = weights.dim() == 2
            B = max(weights.size(0), mean.size(0), cov.size(0)) if is_batched else 1
            K = weights.size(-1)
        else:
            is_batched = mean.dim() == 2
            B = max(mean.size(0), cov.size(0)) if is_batched else 1
            K = 1
        

        squeeze_norm_if_necessary = lambda p : p
        if not is_batched:
            if not is_mixture:
                squeeze_norm_if_necessary = lambda p : p.squeeze_(0, 1)
            else:
                squeeze_norm_if_necessary = lambda p : p.squeeze_(0)
        elif not is_mixture:
            squeeze_norm_if_necessary = lambda p : p.squeeze_(1)

        self.mean, self.cov_chol_decomp, weights = [
            squeeze_norm_if_necessary(p).contiguous() if p is not None else p 
            for p in self.normalize_params(
                self.mean, self.cov_chol_decomp, weights,
                B, K, F, is_mixture, is_batched
            )
        ]
        if is_mixture:
            self.weights_dist = torch.distributions.Categorical(weights)
        else:
            self.weights_dist = None
        
        self.rng = torch.Generator(device=mean.device)
        if seed is not None:
            self.rng.manual_seed(seed)

    def to(self, device : torch.device, seed : Optional[int] = None, set_same_initial_seed : bool = True):
        """
        Moves the GaussianMixture parameters to the specified device.

        This method transfers all internal tensor parameters (mean, covariance
        Cholesky decomposition, and weights_dist if is mixture) to the target
        device and recreates the associated random number generator on that device.

        Args:
            device (torch.device):
                Target device (e.g. ``torch.device("cpu")`` or ``torch.device("cuda")``).
            seed (int, optional):
                Explicit seed to initialize the random number generator on the
                target device. If provided, overrides ``set_same_initial_seed``.
            set_same_initial_seed (bool, default=True):
                If ``True`` and ``seed`` is ``None``, the generator on the target
                device is initialized with the same initial seed as the previous
                generator. This preserves reproducibility across device transfers.

        Returns:
            GaussianMixture:
                The current instance, moved to the specified device.

        Example:
            >>> gm = GaussianMixture(mean, cov, weights, seed=123)
            >>> gm = gm.to(torch.device("cuda"))
            >>> samples = gm.sample(1000)
        """
        self.mean = self.mean.to(device)
        self.cov_chol_decomp = self.cov_chol_decomp.to(device)
        if self.is_mixture:
            self.weights_dist = torch.distributions.Categorical(self.weights_dist.probs.to(device))

        if seed is None and set_same_initial_seed:
            seed = self.rng.initial_seed()

        self.rng = torch.Generator(device=device)

        if seed is not None:
            self.rng.manual_seed(seed)

        return self
    
    @property
    def is_batched(self) -> bool:
        is_mixture = self.is_mixture
        return (not is_mixture and self.mean.dim() == 2) or (is_mixture and len(self.weights_dist.batch_shape) > 0)

    @property
    def is_mixture(self) -> bool:
        return self.weights_dist is not None
    
    @property
    def cov(self) -> torch.Tensor:
        return self.cov_chol_decomp @ self.cov_chol_decomp.mT

    @property
    def F(self) -> int:
        """
        Feature dimensionality of the Gaussian components, i.e. the size of the last dimension of the mean tensor.
        """
        return self.mean.size(-1)

    @property
    def K(self) -> int:
        return self.weights_dist.probs.size(-1) if self.is_mixture else 1

    @property
    def B(self) -> int:
        return self.mean.size(0) if self.is_batched else 1
    
    @property
    def device(self) -> torch.device:
        """
        The device on which the GaussianMixture parameters reside.
        """
        return self.mean.device
    
    @property
    def dtype(self):
        return self.mean.dtype

    def _correction_for_diff(self, diff : torch.Tensor) -> torch.Tensor:
        r"""
        Compute a random correction mask used to adjust rounded component counts so
        that they sum exactly to ``n``.

        This function is used when ``n * weights`` does not sum to exactly ``n`` after
        rounding. Indices are chosen uniformly at random.

        Args:
            diff (Tensor): Scalar integer tensor (shape ``[]``) indicating the required
                adjustment. Positive values add samples to random components, negative
                values subtract.

        Returns:
            Tensor: A correction vector of shape ``(m,)`` with entries in ``{-1, 0, 1}``
            scaled so that the sum equals ``diff``.

        Note:
            This is used only for deterministic mixture sampling.
        """
        idx = torch.randperm(self.K, generator=self.rng, device=diff.device)[:diff.abs()]
        change_mask = torch.zeros(self.K, dtype=diff.dtype).scatter_(0, idx, torch.ones(self.K, dtype=diff.dtype))
        return change_mask * diff.sign()

    def _deterministic_comp_ids(self, n : int):
        r"""
        Compute component indices for deterministic mixture sampling.

        The number of samples per component is given by ``round(n * weights)`` with
        a correction ensuring the sum equals ``n``. Components are then expanded into
        an index tensor used for gathering means and covariances.

        This assumes that self is a mixture (i.e. weights were provided at __init__)

        Args:
            n (int): Number of samples to draw.

        Returns:
            Tensor:
                - If weights are not batched: shape ``(n,)`` containing component indices.
                - If weights are batched: shape ``(b, n)`` with per-batch component indices.

        Raises:
            AssertionError: If internal consistency checks fail.

        """
        rounded_amounts = (n * self.weights_dist.probs).round().to(int)
        diffs_to_total = n - rounded_amounts.sum(dim=-1)
        is_batched = self.is_batched

        if (diffs_to_total != 0).any():
            if is_batched:
                correction = torch.stack([self._correction_for_diff(d) for d in diffs_to_total])
            else:
                correction = self._correction_for_diff(diffs_to_total)
            rounded_amounts += correction

        comp_ids = torch.repeat_interleave(torch.arange(self.B * self.K, device = self.mean.device), rounded_amounts.flatten())
        if is_batched:
            # Reshape per batch and make indices valid
            comp_ids = comp_ids.reshape(self.B, -1) - torch.arange(0, (self.B-1)*self.K + 1, self.K, device = self.mean.device).unsqueeze(1)

        return comp_ids
    
    def _sample_mixture(self, n : int, deterministic_weights : bool = False, reveal_mixture_components : bool = False) -> torch.Tensor:
        r"""
        Sample from the Gaussian mixture model.

        Args:
            n (int): Number of samples to draw.
            deterministic_weights (bool): If ``True``, use deterministic component
                assignments based on rounded mixture weights. Otherwise use multinomial
                sampling via ``Categorical`` distribution.
            reveal_mixture_components (bool): If ``True``, return the component indices
                used for sampling.

        Returns:
            Tensor:
                - For batched mixtures: shape ``(b, n, k)``
                - For unbatched mixtures: shape ``(n, k)``
        """
        if deterministic_weights:
            comp_ids = self._deterministic_comp_ids(n)     # [B, N] or [N]
        else:
            comp_ids = (
                self.weights_dist.sample((n,)) # [N, B] or [N]
                .transpose(-1,0) # [B, N] or [N] - Always valid transpose
            ) 

        if self.is_batched:
            f = self.F
            gathered_means = self.mean.gather(1, comp_ids.unsqueeze(-1).expand(-1, -1, f))
            gathered_decomp_covs = self.cov_chol_decomp.gather(1, comp_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, f, f))
        else:
            gathered_means = self.mean[comp_ids]
            gathered_decomp_covs = self.cov_chol_decomp[comp_ids]


        gaussian_mixture_sample = mvn_random_sample(
            mean = gathered_means,
            cov_chol_decomp = gathered_decomp_covs,
            n = 1,
            rng = self.rng,
            args_checks=False
        )

        mixture_components = comp_ids if reveal_mixture_components else None

        return gaussian_mixture_sample.squeeze(0), mixture_components


    def sample(self, n : int, deterministic_weights : bool = False, reveal_mixture_components : bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Draw samples from the distribution represented by this instance.

        If no ``weights`` were provided at initialization, ``n`` independent samples
        are drawn from each Gaussian in the batch. Otherwise a Gaussian mixture is used.

        Args:
            n (int): Number of samples.
            deterministic_weights (bool): Whether to use deterministic mixture weights.
            reveal_mixture_components (bool): If ``True``, return the component indices
                used for sampling.

        Returns:
            Tensor:
                - If mixture: same as :meth:`sample_mixture`
                - If independent Gaussians without mixture weights:
                    • ``(n, k)`` for unbatched
                    • ``(b, n, k)`` for batched

        """
        if self.is_mixture:
            return self._sample_mixture(n, deterministic_weights, reveal_mixture_components)
        
        sample = mvn_random_sample(
            mean = self.mean,
            cov_chol_decomp = self.cov_chol_decomp,
            n = n,
            rng = self.rng,
            args_checks=False
        )

        if self.B > 1:
            sample = sample.transpose(0,1)

        return sample, None
    
    def manual_seed(self, seed : int) -> None:
        r"""
        Manually set the internal RNG seed.

        Args:
            seed (int): Seed to set for the internal torch.Generator.
        """
        self.rng.manual_seed(seed)

    @staticmethod
    def normalize_params(
        mean: torch.Tensor, cov_chol: torch.Tensor, weights: Optional[torch.Tensor],
        B: int, K: int, F: int, is_mixture: bool, is_batched: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Return mean, cov_chol or cov (and optionally weights) expanded to a
        consistent ``(b, m, ...)`` shape, using ``expand`` (no data copy).

        Returns:
            Tuple of:
                - mean:       ``(b, k, f)``
                - cov_chol:   ``(b, m, f, f)``
                - weights: ``(b, m)`` if mixture, else ``None``
        """
        if not is_mixture and is_batched:
            mean = mean.unsqueeze(1)
            cov_chol = cov_chol.unsqueeze(1)

        mean = mean.expand(B, K, F)
        cov_chol = cov_chol.expand(B, K, F, F)

        if is_mixture:
            assert weights is not None, "Weights cannot be none if is_mixture"
            weights = weights.expand(B, K)
        
        return mean, cov_chol, weights

        

    def _normalized_params(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Return mean, cov_chol (and optionally weights) expanded to a
        consistent ``(b, m, ...)`` shape, using ``expand`` (no data copy).

        Returns:
            Tuple of:
                - mean:       ``(b, k, f)``
                - cov_chol:   ``(b, m, f, f)``
                - weights: ``(b, m)`` if mixture, else ``None``
        """
        is_mixture = self.is_mixture
        return self.normalize_params(
            self.mean, self.cov_chol_decomp, self.weights_dist.probs if is_mixture else None, 
            self.B, self.K, self.F, 
            is_mixture, self.is_batched
        )

    @staticmethod
    def _effective_cov_additive(cov: torch.Tensor, noise_var: float) -> torch.Tensor:
        cov_wn = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * noise_var
        cov_wn = cov_wn.broadcast_to(cov.shape)
        return cov + cov_wn

    def effective_cov_additive(self, noise_var: float) -> torch.Tensor:
        return self._effective_cov_additive(self.cov, noise_var)

    
    def log_prob(self, x: torch.Tensor, check_input: bool = True, white_noise_var: float = 0.0) -> torch.Tensor:
        r"""
        Evaluate the log-probability density at observations ``x``.

        For a mixture, computes:

        .. math::
            \log p(x) = \log \sum_{j=1}^{m} w_j \,\mathcal{N}(x \mid \mu_j, \Sigma_j)

        via the log-sum-exp trick for numerical stability. For independent
        Gaussians (no weights), computes the per-component log-prob directly.

        Optionally, an isotropic white noise term :math:`\sigma^2 I` can be folded
        into each component covariance. This corresponds to the case where
        observations are generated as :math:`x = z + \epsilon` with
        :math:`z \sim \text{GMM}` and :math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I)`,
        which inflates each component covariance to :math:`\Sigma_j + \sigma^2 I`
        while leaving the mixture weights and means unchanged.

        Args:
            x (Tensor): Observations. Either:

                - ``(n, k)``    — ``n`` observations, broadcast over batches if batched.
                - ``(b, n, k)`` — ``n`` batch-specific observations per DGP.

            check_input (bool): If ``True``, validates shape and covariate count of ``x``
                and non-negativity of ``white_noise_var``. Default: ``True``.
            white_noise_var (float): Variance :math:`\sigma^2` of isotropic additive
                noise. If ``0.0`` (default), no inflation is applied. Must be ``>= 0``.

        Returns:
            Tensor:
                - ``(n,)``   if unbatched.
                - ``(b, n)`` if batched.

        Raises:
            AssertionError: If ``check_input=True`` and any shape or value check fails.
        """
        f = self.F
        if check_input:
            if x.size(-1) != f:
                raise AssertionError("x cannot come from current mixture, wrong amount of covariates")
            if x.dim() not in [2,3]:
                raise AssertionError("x has to have shape (b, n, k) or (n, k)")
            if x.dim() == 3 and x.size(0) != self.B:
                raise AssertionError("Wrong batch dimension")
            
            if white_noise_var < 0:
                raise AssertionError("white_noise_var has to be greater equal 0")
            
        mean, cov_chol, weights = self._normalized_params()
        # mean:     (b, k, f)
        # cov_chol: (b, k, f, f)

        # Normalize x to (b, n, f)
        n = x.size(-2)
        x = x.expand(self.B, n, f)
        # x is now (b, n, f)

        # Residuals: (b, n, k, f)
        # x: (b, n, 1, f),  mean: (b, 1, k, f)
        residuals = x.unsqueeze(2) - mean.unsqueeze(1)  # (b, n, k, f)

        # Solve L v = residual for v, then Mahalanobis = ||v||^2
        # L:          (b, m, f, f) -> (b, 1, m, f, f)
        # residuals:  (b, n, k, f) -> (b, n, k, f, 1)
        # Note (x-\mu)^T \Sigma^{-1}(x-\mu) = \|L^{-1}(x-\mu)\|^2_2 =: \|v\|^2_2

        if white_noise_var > 0:
            # Consider the effective covariance including the white noise term: Sigma + sigma^2 I
            cov_normalized = cov_chol @ cov_chol.mT # (b, m, f, f)
            cov_inflated = self._effective_cov_additive(cov_normalized, white_noise_var) # (b, m, f, f)
            cov_chol = torch.linalg.cholesky(cov_inflated) # rewrite cov_chol to the inflated version

        L = cov_chol.unsqueeze(1) # (b, 1, m, f, f)

        r = residuals.unsqueeze(-1)                                     # (b, n, k, f, 1)
        v = torch.linalg.solve_triangular(L, r, upper=False)           # (b, n, k, f, 1)
        mahal = v.squeeze(-1).pow(2).sum(dim=-1)                       # (b, n, k)


        # Log determinant of Sigma from Cholesky diagonal: (b, m)
        log_det = 2.0 * cov_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (b, k)

        # Per-component log-probs: (b, n, k)
        log_norm = -0.5 * (f * log(2 * torch.pi) + log_det)       # (b, k)
        comp_log_probs = log_norm.unsqueeze(1) - 0.5 * mahal           # (b, n, k)

        if self.is_mixture:
            # log_weights: (b, m) -> (b, 1, m)
            log_p = torch.logsumexp(
                comp_log_probs + weights.log().unsqueeze(1), dim=-1
            )                                                           # (b, n)
        else:
            # In this case self.m = m is 1
            log_p = comp_log_probs.squeeze(-1)                        # (b, n)

        if not self.is_batched:
            log_p = log_p.squeeze(0)                                   # (n,)

        return log_p

    def params_str_rep(self, spacing_before : str = ''):
        r"""
        Create a formatted string describing parameter shapes.

        Useful for debugging.

        Args:
            spacing_before (str): Optional indentation prefix.

        Returns:
            str: Formatted description of parameter shapes.
        """
        params_str = spacing_before + f"mean.size = {self.mean.shape}"
        params_str += '\n' + spacing_before + f"cov.size = {self.cov_chol_decomp.shape}"
        if self.is_mixture:
            params_str += '\n' + spacing_before + f"weights.size = {self.weights_dist.probs.shape}"

        return params_str

    ####### Utilities ######## 
        
    @staticmethod
    def dist_params_check(
        mean : torch.Tensor, 
        cov : torch.Tensor, 
        weights : Optional[torch.Tensor] = None,
        symmetry_rtol_atol : Tuple[float] = [0.0, 0.0]
    ) -> None:
        r"""
        Validate all distribution parameters for shape and consistency.

        Checks:
            - mean and covariance dimensionality
            - covariance symmetry
            - consistency of m-axis across parameters
            - weight normalization
            - matching Gaussian dimensionality ``k``

        Args:
            mean (Tensor): Mean tensor of shape ``(f,)``, ``(k, f)``, ``(b, f)`` or ``(b, k, f)``.
            cov (Tensor): Covariance tensor of shape ``(..., f, f)``.
            weights (Tensor, optional): Mixture weights of shape ``(m,)`` or ``(b, m)``.
            symmetry_rtol_atol (Tuple[float]): Tolerances for symmetry check.

        Raises:
            AssertionError: If any parameter check fails.
        """
        weights_is_not_none = weights is not None
        if not (mean.dim() in (1, 2) or (mean.dim() == 3 and weights_is_not_none)):
             raise AssertionError("Means  needs to be of shape (f,), (k, f) or (b, f) or (b, k, f) - in which case weights are necessary")
        
        if not (cov.dim() == mean.dim()+1):
            raise AssertionError("Covs has to have one more dimension than means")
        if not (mean.dtype == cov.dtype):
            raise AssertionError("Means and covs need to have the same dtype")
        if not (mean.device == cov.device):
            raise AssertionError("Means and covs need to have the same device")
        
        if weights_is_not_none:
            if not (weights.dim() == mean.dim() - 1):
                raise AssertionError("weights must have one dimension less than means")
            if not (weights.dtype == mean.dtype):
                raise AssertionError("weights must have same dtype as mean and cov")
            if not (weights.device == mean.device):
                raise AssertionError("weights must have the same device as mean and cov")
            
            m = weights.size(-1)
            size_checks = mean.size(-2) in (1, m) and cov.size(-3) in (1, m)
            if not size_checks:
                raise AssertionError("mean or cov size does not match weights size, broadcasting rules violated")

            error_from_floating_point_operation = torch.finfo(weights.dtype).eps / 2
            tol = (weights.size(-1) - 1) * error_from_floating_point_operation
            weights_sum = weights.sum(axis=-1)
            if not torch.allclose(weights_sum, torch.ones_like(weights_sum), 
                                  atol = tol, rtol = 0):
                raise AssertionError("All weights per batch need to add up to 1")
        
        if not (mean.size(-1) == cov.size(-1) == cov.size(-2)):
            raise AssertionError("Covariate count k is not constant")
        if not torch.allclose(cov, cov.transpose(-1, -2), *symmetry_rtol_atol):
            raise AssertionError("Covariance matrix not symmetric")

    def univariate_mixture_as_tex(
        self, 
        suffix: str, 
        letter_for_data: str = 'X', 
        start_idx_mixtures: int = 0,
        weights_iter_symbol: str = "k"
    ) -> str:
        mus, covs_chol_decomp, weights = self._normalized_params()
        covs = covs_chol_decomp @ covs_chol_decomp.mT
        is_mixture = self.is_mixture
        is_batched = self.is_batched

        tex_objs = []
        s = suffix

        def letter_maker(suffix, add_to_s = ""):
            full_subind = suffix + add_to_s
            if len(full_subind) > 0:
                return letter_for_data + "_{" + full_subind + r"}"
            
            return letter_for_data

        def mvn_latex(mu, cov, add_to_s = ""):
            mu_vec = (
                "\\begin{bmatrix}\n\t" +
                "\\\\\n\t".join([f"{mu_k:.1f}" for mu_k in mu]) +
                "\n\\end{bmatrix}"
            )
            Sigma_vcov = (
                "\\begin{bmatrix}\n\t" +
                "\\\\\n\t".join([" & ".join([f"{c:.1f}" for c in row]) for row in cov]) +
                "\n\\end{bmatrix}"
            )

            return (
                letter_maker(s, add_to_s) + r" \sim" + 
                r"\mathcal{N}\left(\mu_{" + s + add_to_s + "} = " + mu_vec +
                r", \Sigma_{" + s + add_to_s + "} = " + Sigma_vcov + r"\right)"
            )
        
        wi = weights_iter_symbol

        for b_idx, (mu_b, cov_b) in enumerate(zip(mus, covs)):
            if is_mixture:
                tex_objs.append([])
                current_weights = []
            for m_idx, (mu_m, cov_m) in enumerate(zip(mu_b, cov_b)):
                if is_mixture:
                    current_weights.append(weights[b_idx, m_idx] if is_batched else weights[0, m_idx])
                    m_idx += start_idx_mixtures
                    tex_objs[b_idx].append(mvn_latex(mu_m, cov_m, add_to_s=f"_{{{m_idx}}}"))
                else:
                    tex_objs.append(mvn_latex(mu_m, cov_m))
                    #break - not necessary, there will be only one iteration

            if is_mixture:
                subind = s + '_{' + wi + '}'
                tex_objs[b_idx].append(
                    letter_maker(s) + rf" \mid \{{Z = {wi} \}}  \sim \mathcal{{N}}\left(\mu_{{{subind}}}, \Sigma_{{{subind}}}\right)," +
                    r" \, Z \sim \text{{Categorical}}\left(" +
                        ", ".join([fr"\pi_{{{k}}}={w:.2f}" for k, w in zip(range(start_idx_mixtures, start_idx_mixtures+self.K), current_weights)]) +
                    r"\right)"
                )

                
        dist_strs = []
        for text_obj in tex_objs:
            if is_mixture:
                vars_dist_str = "$$\n" + r", \;".join(text_obj[:-1]) + "\n$$"
                gm_dist_str = "$$\n" + text_obj[-1] + "\n$$"

                dist_strs.append(gm_dist_str + "\nwith\n" + vars_dist_str)
            else:
                dist_strs.append("$$\n" + text_obj + "\n$$")

        

        return text_obj, dist_strs
    
def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / sqrt(2)))

def bayes_rate_two_class_mvn_gaussian_equal_cov(
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        cov_chol: torch.Tensor,
        p1: float,
        odds_factor: float = 1.0
    ) -> torch.Tensor:
    """
    Based on the formula by Ripley (1996, 2nd Edition) Chapter 2, Page 22.
    He calls it probability of missclassification
    """
    mu_diff = mu1 - mu2
    mu_diff.unsqueeze_(-1)

    delta = torch.linalg.norm(
        torch.linalg.solve_triangular(cov_chol, mu_diff, upper=False).squeeze(-1),
        ord=2,
        dim=-1
    )
    # Cache results for numeric stability
    minus_half_delta = -0.5 * delta
    p2 = 1 - p1
    odds = p2/p1
    log_prob_ratio = log(odds*odds_factor)
    ratio_div_delta = log_prob_ratio / delta


    pmc_1 = p1 * standard_normal_cdf(
        minus_half_delta + ratio_div_delta
    )
    pmc_2 = p2 * standard_normal_cdf(
        minus_half_delta - ratio_div_delta
    )

    return pmc_1 + pmc_2



if __name__ == "__main__":
    batch_size, count_mixtures, count_covariates = 4, 4, 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    rng = torch.Generator(device)

    functionality_checks = []

    print("Step 1: checking functionality of eigen_decomp_proj_to_pd")
    before_proj_to_pd = torch.randn((count_covariates, count_covariates))

    after_proj_to_pd = eigen_decomp_proj_to_pd(before_proj_to_pd)

    functionality_checks.append(before_proj_to_pd.shape == after_proj_to_pd.shape)

    print("\tShape mantained:", functionality_checks[-1])

    functionality_checks.append((torch.linalg.eigvalsh(after_proj_to_pd) >=0).all().item())
    print("\tResult is positive definite:",functionality_checks[-1])

    print("\nStep 2: Checking if matrices can be constructed with random vcovs")
    batch_size, count_mixtures, count_covariates = 4, 4, 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    rng = torch.Generator(device)

    all_covs = torch.stack([torch.stack([random_vcov_matrix(f=count_covariates, generator=rng) for _ in range(count_mixtures)]) for _ in range(batch_size)])
    functionality_checks.append(all_covs.shape == torch.Size([batch_size, count_mixtures, count_covariates, count_covariates]))
    print("\tall_covs.size is as expected:", functionality_checks[-1])
    functionality_checks.append((all_covs.transpose(-1,-2) == all_covs).all().item())
    print("\tGenerated covs are symmetric:", functionality_checks[-1])
    functionality_checks.append((torch.linalg.eigvalsh(all_covs) > 0).all().item())
    print("\tGenerated covs are positive definite:", functionality_checks[-1])

    print("\n\nStep 3: Attempts to create and sample with different shapes of params the GaussianMixture\n")

    #Generate the rest of args
    mu = torch.randn((batch_size, count_mixtures, count_covariates), generator=rng)

    weights = torch.rand((batch_size, count_mixtures), generator = rng)
    weights /= weights.sum(axis=-1, keepdim=True)

    n = 100

    batched_size = torch.Size([batch_size, n, count_covariates])
    unbatched_size = torch.Size([n, count_covariates])

    trial_elements = [
        (
            "no weights, " + ("un" if unbat else "") + "batched params",
            True,
            {
                "mean" : mu[0, 0] if unbat else mu[:, 0],
                "cov" : all_covs[0, 0] if unbat else all_covs[:, 0],
                "weights" : None
            },
            unbatched_size if unbat else batched_size
        ) for unbat in [True, False]] + sum([[
        (
            "weigths, " + ("" if bat else "un") + "batched params, weight based " + ("deterministic" if det else "random") +" mvn sampling",
            det,
            {
                "mean" : mu if bat else mu[0],
                "cov" : all_covs if bat else all_covs[0],
                "weights" : weights if bat else weights[0]
            },
            batched_size if bat else unbatched_size
        )
    for det in [True, False]] for bat in [True, False]], [])

    for num_trial, (msg, deterministic_weights, mixture_kwargs, expected_size) in enumerate(trial_elements):
        print(f"\nAttempt {num_trial + 1}: {msg}")
        dist = GaussianMixture(**mixture_kwargs)
        print("Params shapes:")
        print(dist.params_str_rep('  '), "\n")

        sample = dist.sample(n, deterministic_weights)
        print("\tExpected size:", expected_size)
        print("\tRealized size:", sample.shape)
        functionality_checks.append(sample.shape == expected_size)
        print("\tExpectation realized:", functionality_checks[-1], "\n")

    print("All checks were passed:", all(functionality_checks))