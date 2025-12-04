from typing import Optional, Tuple
import torch
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

if __name__ == "__main__":
    print("Small test to check functionality")
    print("\nFirst test: for unbatched\n")
    count_covariates = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    rng = torch.Generator(device)
    mu = torch.zeros(count_covariates)
    sigma_bad, sigma_good = generate_sigma_bad_and_good(k = count_covariates, proportion_var_dif=1.0, generator = rng, device=device, dtype= torch.get_default_dtype())
    print("generate_sigma_bad_and_good succesful, therefore random_vcov_matrix and eigen_decomp_proj_to_pd as well")
    n=100
    sample = mvn_random_sample(mu, torch.linalg.cholesky(sigma_bad), n, rng)
    print("mvn_random_sample succesful as well")
    print(f"\tExpected sample shape: [{n}, {count_covariates}]")
    print("\tSample shape:", sample.shape)

    print("\nFirst test: for batched\n")
    batch_size = 2
    mu = torch.randn((batch_size,count_covariates))
    sample = mvn_random_sample(
        mu,
        torch.linalg.cholesky(torch.stack((sigma_bad, sigma_good))),
        n,
        rng
    )
    print("Succesful this case too")
    print(f"\tExpected sample shape: [{n}, {batch_size}, {count_covariates}]")
    print("\tSample shape:", sample.shape)

    