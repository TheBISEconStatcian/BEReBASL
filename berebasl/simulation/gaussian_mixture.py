from math import log
import torch
from warnings import warn

from typing import Dict, Optional, Tuple, Union


def random_vcov_matrix(
        k: int,
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
        k (int): Dimension of the covariance matrix.
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
        torch.Tensor: A symmetric, positive definite covariance matrix of shape ``(k, k)``.

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
        A = torch.randn((k, k), generator=generator, dtype=dtype, device=device) # random normal matrix, sparser correlations for high k
    else:
        A = 2*torch.rand((k,k), generator=generator, dtype = dtype, device=device) - 1 # random uniform matrix, correlations closer to 0, the higher k

    # Step 2: Define correlation matrix from base sampling
    Q = A @ A.T # Make sure of symmetry while using full randomness
    D = torch.sqrt(torch.diag(Q)) # Help vector for normalization
    corr_mat = Q / torch.outer(D, D) # corr_mat[i, j] = cosine_similarity(A[i], A[j]), so range [-1, 1] guaranteed

    # Step 3: Rescale corr_mat with sampled variances
    ## Variance sampling from uniform distribution
    variances = torch.rand(k, generator=generator, dtype = dtype, device=device) * (var_range[1] - var_range[0]) + var_range[0]
    ## Rescaling via outer prouct of standard deviations
    stds = torch.sqrt(variances)
    norm_factors_pearson_corr = torch.outer(stds, stds) # guaranteed to be symmetric, denominators of pearson correlation
    vcov = corr_mat * norm_factors_pearson_corr

    # Step 4: Avoid semi positive definitness of the matrix
    vcov = vcov + eps * torch.eye(k, device=device, dtype=dtype)

    # Step 5: Ensure **exact** symmetry without compromising randomness
    i, j = torch.tril_indices(k, k, offset=-1)
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
        mat (torch.Tensor): Input square matrix of shape ``(k, k)``.
        eps (float, optional): Minimum eigenvalue threshold to enforce positive definiteness.
            Defaults to ``1e-6``.
        ensure_symmetry (bool, optional): If True, symmetrize the input before decomposition.
            Defaults to False.

    Returns:
        torch.Tensor: Symmetric positive semidefinite matrix of shape ``(k, k)``.

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
        mean (torch.Tensor): Location parameter of a MVN. Shape ``(k,)`` or ``(b, k)`` or ``(1,5)``.
        cov_chol_decomp (torch.Tensor): Variance-Covariance matrix of MVN after cholesky decomposition. 
            Shape ``(k,k)``` or ``(b, k, k)`` or ``(1, k, k)``, ``cov.dim()==mean.dim()+1`` should hold.
        n (int): Count of vectors to be sampled (per batch).
        rng (Optional[torch.Generator]): If passed, sampling is done using this
            generator.
        args_checks (bool): If true, it will be checked whether the shapes of mean and
            cov are as expected, whether symmetry (w. r. t. to the last wo dims for each beach)
            is given within the range of ``symmetry_rtol_atol`` for ``cov`` and type checks
            are done for ``n`` and ``rng``.`
        symmetry_rtol_atol (Tuple[float,float]): Corresponds to the (rtol, a_tol) parameters
            of ``torch.allclose``, passed as ``*args``, so ordering is important. Ignored if
            ``not args_checks``.
    Returns:
        torch.Tensor:
            A tensor of shape ``(n, k)`` or ``(n, b, k)`` containing the ``n`` sampled vectors (for each batch).

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
        assert (mean.size(-1) == cov_chol_decomp.size(-1)) and (cov_chol_decomp.size(-1) == cov_chol_decomp.size(-2)), "mean must have shape [k] and cov shape [k, k]"
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
        mean (Tensor): Mean tensor of shape ``(k,)``, ``(m, k)``, ``(b, k)`` or ``(b, m, k)``.
            Supports both batched and unbatched mixtures.
        cov (Tensor): Covariance tensor of shape ``(k, k)``, ``(m, k, k)``, ``(b, k, k)`` or ``(b, m, k, k)``.
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
        self.mean = mean
        self.cov_chol_decomp = torch.linalg.cholesky(cov)
        self.m = 1
        self.is_mixture = weights is not None
        if self.is_mixture:
            self.weights_are_batched = weights.dim() == 2
            if self.weights_are_batched:
                self.b = weights.size(0)
            else:
                self.b = 1
            self.m =  weights.size(-1)
            self.weights_dist = torch.distributions.Categorical(weights)
        else:
            self.b = 1 if self.mean.dim() == 1 else self.mean.size(1)

        self.is_batched = self.b>1
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
    def cov(self):
        return self.cov_chol_decomp @ self.cov_chol_decomp.mT

    @property
    def k(self):
        return self.mean.size(-1)
    
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
        idx = torch.randperm(self.m, generator=self.rng, device=diff.device)[:diff.abs()]
        change_mask = torch.zeros(self.m, dtype=diff.dtype).scatter_(0, idx, torch.ones(self.m, dtype=diff.dtype))
        return change_mask * diff.sign()

    def deterministic_comp_ids(self, n : int):
        r"""
        Compute component indices for deterministic mixture sampling.

        The number of samples per component is given by ``round(n * weights)`` with
        a correction ensuring the sum equals ``n``. Components are then expanded into
        an index tensor used for gathering means and covariances.

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
        if (diffs_to_total != 0).any():
            if self.weights_are_batched:
                correction = torch.stack([self._correction_for_diff(d) for d in diffs_to_total])
            else:
                correction = self._correction_for_diff(diffs_to_total)
            rounded_amounts += correction

        comp_ids = torch.repeat_interleave(torch.arange(self.b * self.m, device = self.mean.device), rounded_amounts.flatten())
        if self.weights_are_batched:
            # Reshape per batch and make indices valid
            comp_ids = comp_ids.reshape(self.b, -1) - torch.arange(0, (self.b-1)*self.m + 1, self.m, device = self.mean.device).unsqueeze(1)

        return comp_ids
    
    def sample_mixture(self, n : int, deterministic_weights : bool = False):
        r"""
        Sample from the Gaussian mixture model.

        Args:
            n (int): Number of samples to draw.
            deterministic_weights (bool): If ``True``, use deterministic component
                assignments based on rounded mixture weights. Otherwise use multinomial
                sampling via ``Categorical`` distribution.

        Returns:
            Tensor:
                - For batched mixtures: shape ``(b, n, k)``
                - For unbatched mixtures: shape ``(n, k)``
        """
        if deterministic_weights:
            comp_ids = self.deterministic_comp_ids(n)
        else:
            comp_ids = self.weights_dist.sample((n,)).transpose(-1,0) # Always valid transpose

        if self.weights_are_batched:
            k = self.mean.size(-1)
            gathered_means = self.mean.gather(1, comp_ids.unsqueeze(-1).expand(-1, -1, k))
            gathered_decomp_covs = self.cov_chol_decomp.gather(1, comp_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, k))
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

        return gaussian_mixture_sample.squeeze(0)


    def sample(self, n : int, deterministic_weights : bool = False):
        r"""
        Draw samples from the distribution represented by this instance.

        If no ``weights`` were provided at initialization, ``n`` independent samples
        are drawn from each Gaussian in the batch. Otherwise a Gaussian mixture is used.

        Args:
            n (int): Number of samples.
            deterministic_weights (bool): Whether to use deterministic mixture weights.

        Returns:
            Tensor:
                - If mixture: same as :meth:`sample_mixture`
                - If independent Gaussians without mixture weights:
                    • ``(n, k)`` for unbatched
                    • ``(b, n, k)`` for batched

        """
        if self.is_mixture:
            return self.sample_mixture(n, deterministic_weights)
        
        sample = mvn_random_sample(
            mean = self.mean,
            cov_chol_decomp = self.cov_chol_decomp,
            n = n,
            rng = self.rng,
            args_checks=False
        )

        if self.b > 1:
            sample = sample.transpose(0,1)

        return sample
    
    def manual_seed(self, seed : int) -> None:
        r"""
        Manually set the internal RNG seed.

        Args:
            seed (int): Seed to set for the internal torch.Generator.
        """
        self.rng.manual_seed(seed)

    def _normalized_params(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Return mean, cov_chol (and optionally log_weights) expanded to a
        consistent ``(b, m, ...)`` shape, using ``expand`` (no data copy).

        Returns:
            Tuple of:
                - mean:       ``(b, m, k)``
                - cov_chol:   ``(b, m, k, k)``
                - weights: ``(b, m)`` if mixture, else ``None``
        """
        k = self.mean.size(-1)

        if not self.is_mixture and self.is_batched:
            mean = self.mean.squeeze(1)
            cov_chol = self.cov_chol_decomp.squeeze(1)
        else:
            mean = self.mean.expand(self.b, self.m, k)
            cov_chol = self.cov_chol_decomp.expand(self.b, self.m, k, k)
        

        weights = None
        if self.is_mixture:
            weights = self.weights_dist.probs.expand(self.b, self.m)
        else:
            if self.is_batched:
                mean = self.mean.squeeze(1)
                cov_chol = self.cov_chol_decomp.squeeze(1)



        return mean, cov_chol, weights
    
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
        k = self.k
        if check_input:
            if x.size(-1) != k:
                raise AssertionError("x cannot come from current mixture, wrong amount of covariates")
            if x.dim() not in [2,3]:
                raise AssertionError("x has to have shape (b, n, k) or (n, k)")
            if x.dim() == 3 and x.size(0) != self.b:
                raise AssertionError("Wrong batch dimension")
            
            if white_noise_var < 0:
                raise AssertionError("white_noise_var has to be greater equal 0")
            
        mean, cov_chol, weights = self._normalized_params()
        # mean:     (b, m, k)
        # cov_chol: (b, m, k, k)

        # Normalize x to (b, n, k)
        n = x.size(1)
        x = x.expand(self.b, n, k)
        # x is now (b, n, k)

        # Residuals: (b, n, m, k)
        # x: (b, n, 1, k),  mean: (b, 1, m, k)
        residuals = x.unsqueeze(2) - mean.unsqueeze(1)  # (b, n, m, k)

        # Solve L v = residual for v, then Mahalanobis = ||v||^2
        # L:          (b, m, k, k) -> (b, 1, m, k, k)
        # residuals:  (b, n, m, k) -> (b, n, m, k, 1)
        # Note (x-\mu)^T \Sigma^{-1}(x-\mu) = \|L^{-1}(x-\mu)\|^2_2 =: \|v\|^2_2

        if white_noise_var == 0.0:
            # Simple case: no white noise
            L = cov_chol.unsqueeze(1)                                       # (b, 1, m, k, k)
        else:
            # Harder case: need to add the noise manually
            cov_normalized = cov_chol @ cov_chol.mT # (b, m, k, k)
            cov_white_noise = white_noise_var * torch.eye(k, device=self.device, dtype=self.dtype) # [k, k]
            cov_wn_normalized = cov_white_noise.expand(1,1,k,k) # [1,1, k, k]
            cov_inflated = cov_normalized + cov_wn_normalized # (b, m, k, k)
            L = torch.linalg.cholesky(cov_inflated).unsqueeze(1) # (b, 1, m, k, k)

        r = residuals.unsqueeze(-1)                                     # (b, n, m, k, 1)
        v = torch.linalg.solve_triangular(L, r, upper=False)           # (b, n, m, k, 1)
        mahal = v.squeeze(-1).pow(2).sum(dim=-1)                       # (b, n, m)


        # Log determinant of Sigma from Cholesky diagonal: (b, m)
        log_det = 2.0 * cov_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (b, m)

        # Per-component log-probs: (b, n, m)
        log_norm = -0.5 * (k * log(2 * torch.pi) + log_det)       # (b, m)
        comp_log_probs = log_norm.unsqueeze(1) - 0.5 * mahal           # (b, n, m)

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
            mean (Tensor): Mean tensor of shape ``(k,)``, ``(m, k)``, ``(b, k)`` or ``(b, m, k)``.
            cov (Tensor): Covariance tensor of shape ``(..., k, k)``.
            weights (Tensor, optional): Mixture weights of shape ``(m,)`` or ``(b, m)``.
            symmetry_rtol_atol (Tuple[float]): Tolerances for symmetry check.

        Raises:
            AssertionError: If any parameter check fails.
        """
        assert mean.dim() in (1,2,3), "Means  needs to be of shape (k,), (m, k) or (b, k) or (b, m, k)"
        assert cov.dim() == mean.dim()+1, "Covs has to have one more dimension than means"
        assert mean.dtype == cov.dtype, "Means and covs need to have the same dtype"
        assert mean.device == cov.device, "Means and covs need to have the same device"
        
        if weights is not None:
            assert weights.dim() == mean.dim() - 1, "weights must have one dimension less than means"
            assert weights.dtype == mean.dtype, "weights must have same dtype as mean and cov"
            assert weights.device == mean.device, "weights must have the same device as mean and cov"

            size_checks = [
                (mean.size(-2) == 1) and (cov.size(-3) == weights.size(-1)),
                (mean.size(-2) == weights.size(-1)) and (cov.size(-3) == 1),
                mean.size(-2) == cov.size(-3) == weights.size(-1)
            ]
            assert any(size_checks), "Non constant m-axis"

            error_from_floating_point_operation = torch.finfo(weights.dtype).eps / 2
            tol = (weights.size(-1) - 1) * error_from_floating_point_operation
            weights_sum = weights.sum(axis=-1)
            assert torch.allclose(weights_sum, torch.ones(weights_sum.shape, dtype=weights.dtype), 
                                  atol = tol, rtol = 0), "All weights per batch need to add up to 1"
        
        assert mean.size(-1) == cov.size(-1) == cov.size(-2), "Covariate count k is not constant"
        assert torch.allclose(cov, cov.transpose(-1, -2), *symmetry_rtol_atol), "Covariance matrix not symmetric"


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

    all_covs = torch.stack([torch.stack([random_vcov_matrix(k=count_covariates, generator=rng) for _ in range(count_mixtures)]) for _ in range(batch_size)])
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