from copy import deepcopy
from contextlib import contextmanager
import inspect
from math import log, sqrt

from numpy import ndarray
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Dict, Optional, Tuple, Union

from berebasl.utils.tensor_validation import assert_tensors
from berebasl.simulation.credit_data_simulation import CreditDataGenerator
from berebasl.simulation.gaussian_mixture import GaussianMixture

def function_has_expected_signature(fun : Callable, count_params : int):
    sig = inspect.signature(fun)

    parameter_count = len(sig.parameters)

    if parameter_count < count_params:
        return False
    
    if parameter_count == count_params:
        return True
    
    return sum([p.default is inspect._empty for p in sig.parameters.values()]) <= count_params

class Classifier:
    def __init__(
            self,
            model,
            predict_model_probs : Callable[["model", torch.Tensor], torch.Tensor],
            train_model : Callable[["model", torch.Tensor, torch.Tensor], None],
            reset_parameters_to_initial_state : Callable[["model"], None],
            copy_current_state_dict : Callable[["model"], dict],
            state_dict_loader : Callable[[dict], None]
    ):
        self.model = model
        self.predict_model_probs = predict_model_probs
        self.train_model = train_model
        self.reset_parameters_to_initial_state = reset_parameters_to_initial_state
        self.copy_current_state_dict = copy_current_state_dict
        self.state_dict_loader = state_dict_loader
        

    def fit(self, features : torch.Tensor, labels : torch.Tensor) -> None:
        self.train_model(self.model, features, labels)

    def predict_proba(self, features : torch.Tensor) -> torch.Tensor:
        # with features.shape = [..., k], it should
        # return a tensor of shape [..., 2] containing the predicted
        # probabilities of label = 0, 1 respectively along the last axis
        return self.predict_model_probs(self.model, features)
    
    def reset_parameters_to_initial(self):
        self.reset_parameters_to_initial_state(self.model)

    def load_from_state_dict(self, state_dict : dict, *args, **kwargs):
        self.state_dict_loader(state_dict, *args, **kwargs)

    def to_state_dict(self):
        return self.copy_current_state_dict(self.model)

    @staticmethod
    def obj_has_needed_funs(obj):
        expected_funs_with_param_count = [
            ("fit", 2),
            ("predict_proba", 1),
            ("reset_parameters_to_initial", 0)
        ]
        for fun_name, param_count in expected_funs_with_param_count:
            if not hasattr(obj, fun_name):
                return False
            
            if not function_has_expected_signature(getattr(obj, fun_name), param_count):
                return False
            
        return True

class TorchLogistic(nn.Module):
    r"""
    Shallow logistic regression model with a single linear estimator and
    probability mapping for binary or multiclass classification.

    The class allows to mimic the behavior of classical GLM-style maximum
    likelihood estimation, including the binary logit model and the multinomial
    softmax model by using full-batch L-BFGS optimization in the ``fit`` method.

    The model supports inputs of arbitrary leading shape ``[...]`` as long as the
    final dimension corresponds to ``n_features``. All leading dimensions are treated
    as part of a single flattened batch during optimization through the ``fit`` method..

    Attributes
    ----------
    lin_estimator : nn.Linear
        Linear predictor mapping ``n_features`` to ``n_classes``.
    n_classes : int
        Number of target classes. ``2`` selects the binary formulation.
    lbfgs_kwargs : dict
        Keyword arguments passed to the L-BFGS optimizer.
    logit_to_probs : Callable[[torch.Tensor], torch.Tensor]
        Function converting logits to class probabilities. Set at
        initialization based on ``n_classes``.
    W_init : torch.Tensor
        buffer of shape ``[n_classes , n_features]`` if multiclass otherwise
        ``[1 , n_features]`` containing the weights after initialization as
        done in ``__init__``
    b_init: torch.Tensor
        buffer of shape ``[n_classes]`` if multiclass otherwise
        ``[1]`` after initialization as done in ``__init__``

    Methods
    -------
    fit(X, y, reduction="sum")
        Fit the model using full-batch L-BFGS.
    predict_proba(X)
        Compute class probabilities.
    predict(X)
        Return class predictions via ``argmax`` over probabilities.
    reset_parameters()
        Restore the linear layer to its stored initial parameters.
    _binary_probs(logits)
        Convert binary logits to probabilities.
    _multiclass_probs(logits)
        Convert multiclass logits to probabilities.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        lbfgs_kwargs: Optional[dict] = None,
        seed_for_weight_init: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        secure_init: bool = True,
    ):
        r"""
        Initialize the logistic regression model.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_classes : int, default=2
            Number of output classes. If ``2``, the model uses the binary logistic
            formulation with a single logit :math:`z = \log \frac{p(y=1 \mid x)}{p(y=0 \mid x)}`.
            For ``n_classes > 2``, the model outputs ``n_classes`` logits.
        lbfgs_kwargs : dict, optional
            Keyword arguments forwarded to ``torch.optim.LBFGS``. If ``None``,
            a default configuration is used. When ``secure_init=True``, the keys
            are validated against the optimizer's constructor.
        seed_for_weight_init : int, optional
            Seed used to initialize a dedicated ``torch.Generator`` for deterministic
            parameter initialization. If ``None``, initialization remains deterministic
            with respect to the created generator inside the method but does not fix a seed.
        device : torch.device, optional
            Device on which parameters and buffers are allocated.
        dtype : torch.dtype, optional
            Data type for model parameters.
        secure_init : bool, default=True
            If ``True``, validates ``lbfgs_kwargs`` and enforces ``n_classes >= 2``.

        Notes
        -----
        The initial parameters of the linear estimator are created using the provided
        generator and stored as buffers (``W_init`` and ``b_init``). They can be
        restored exactly via ``reset_parameters_to_initial``. The mapping from logits
        to probabilities is selected once at initialization and stored in
        ``self.logit_to_probs`` to avoid branching during inference.
        """

        super().__init__()
        self.n_classes = int(n_classes)
        if lbfgs_kwargs is None:
            lbfgs_kwargs = {
                'lr' : 1,
                'max_iter' :  100,
                'tolerance_grad' :  torch.finfo(torch.get_default_dtype()).eps ** (2/3),
                'tolerance_change': torch.finfo(torch.get_default_dtype()).eps ** (7/8),
                'history_size' : 50,
                'line_search_fn' :  'strong_wolfe'
            }
        if secure_init:
            if self.n_classes < 2:
                raise ValueError("n_classes needs to be 2 or bigger")
            allowed_params = {k for k in inspect.signature(optim.LBFGS.__init__).parameters if k not in ("self", "params")}
            kwargs_ok = set(lbfgs_kwargs).issubset(allowed_params)
            if not kwargs_ok:
                raise ValueError("At least one of the keys in lbfgs_kwargs doesn't correspond to a named kwarg of optim.LBFGS.__init__")
            self.lbfgs_kwargs = lbfgs_kwargs
        n_logits_output = 1 if self.n_classes == 2 else self.n_classes

        self.lin_estimator = nn.Linear(
            in_features=n_features,
            out_features=n_logits_output,
            bias=True,
            dtype=dtype,
            device=device
        )

        rng = torch.Generator(device=device)
        if seed_for_weight_init is not None:
            rng = rng.manual_seed(seed_for_weight_init)

        self.reset_parameters(rng)

        self.register_buffer("W_init", self.lin_estimator.weight.detach().clone()) 
        self.register_buffer("b_init", self.lin_estimator.bias.detach().clone())

        if n_classes == 2: 
            self.logit_to_probs = self._binary_probs 
        else: 
            self.logit_to_probs = self._multiclass_probs

    @property
    def n_features(self) -> int:
        return self.lin_estimator.in_features
    
    @staticmethod
    def _binary_probs(logits: torch.Tensor) -> torch.Tensor:
        r"""
        Convert binary logits to probabilities.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape ``[...]`` representing the score for
            the positive class.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``[..., 2]`` with columns
            ``[p(y=0), p(y=1)]``.

        Notes
        -----
        Probabilities are computed via the sigmoid function:

        .. math::

            p(y=1 \mid x) = \sigma(z), \qquad
            p(y=0 \mid x) = 1 - \sigma(z)
        """
        p1 = torch.sigmoid(logits).unsqueeze(-1)
        p0 = 1 - p1
        return torch.cat([p0, p1], dim=-1)


    @staticmethod
    def _multiclass_probs(logits: torch.Tensor) -> torch.Tensor:
        r"""
        Convert multiclass logits to probabilities.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape ``[..., n_classes]``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``[..., n_classes]`` obtained via softmax.

        Notes
        -----
        The softmax function normalizes logits into a valid categorical
        distribution:

        .. math::

            p(y=k \mid x) =
            \frac{\exp(z_k)}{\sum_j \exp(z_j)}
        """
        return torch.softmax(logits, dim=-1)


    def forward(self, X : torch.Tensor) -> torch.Tensor:
        r"""
        Compute raw logits from input features.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``[..., n_features]``. All leading dimensions are
            preserved.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[...]`` for binary classification or
            ``[..., n_classes]`` for multinomial classification.
        """
        # X has shape [..., n_features]
        logits = self.lin_estimator(X) # [..., n_logits_output]

        return logits.squeeze(-1) # [...] if self.n_classes == 2 else [..., n_features]
    
    def predict_proba(self, X : torch.Tensor) -> torch.Tensor:
        r"""
        Compute class probabilities.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``[..., n_features]``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``[..., n_classes]``. For binary classification,
            the output is ``[..., 2]`` with columns ``[p(y=0), p(y=1)]``.

        Notes
        -----
        For ``n_classes == 2``, probabilities are computed via the sigmoid function:

        .. math::

            p(y=1 \mid x) = \sigma(z), \qquad p(y=0 \mid x) = 1 - \sigma(z)

        For ``n_classes > 2``, probabilities are computed via softmax.
        """
        # X has shape [..., n_features]
        logits = self.forward(X) # [..., n_logits_output]
        return self.logit_to_probs(logits) # [..., n_classes]
    
    def predict(self, X) -> torch.Tensor:
        r"""
        Predict the most likely class.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``[..., n_features]``.

        Returns
        -------
        torch.Tensor
            Integer class labels of shape ``[...]`` (same leading dimensions as ``X``).
        """
        # X has shape [..., n_features]
        probs = self.predict_proba(X) # [..., n_classes]
        return torch.argmax(probs, dim=-1) # [...] (same as the other dims of X)
    
    def decision_function(self, X : torch.Tensor) -> torch.Tensor:
        r"""
        Compute the raw decision scores (logits) before any nonlinear activation.

        This method returns the model's linear predictor:
        
        - For binary logistic regression (``n_classes == 2``), this is a single logit
          ``z`` representing the log-odds

          .. math::

              z = \log \frac{p(y=1 \mid x)}{p(y=0 \mid x)}.

        - For multinomial logistic regression (``n_classes > 2``), this is a vector of
          logits ``z_k`` for each class, prior to the softmax transformation.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``[..., n_features]``. All leading dimensions are
            preserved.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``[..., 1]`` for binary classification or
            ``[..., n_classes]`` for multinomial classification.

        Notes
        -----
        - This method is equivalent to calling :meth:`forward`. It is provided for
          compatibility with scikit-learn and classical GLM terminology, where the
          linear predictor is often inspected directly.
        - The output of ``decision_function`` is typically used for ROC curves,
          precision-recall curves, calibration analysis, and threshold tuning, since
          it exposes the model's unnormalized decision scores.
        """
        return self.forward(X)
    
    def reset_parameters_to_initial(self):
        r"""
        Restore the linear estimator to its stored initial parameters.

        Notes
        -----
        Copies ``W_init`` and ``b_init`` back into ``lin_estimator``. This provides
        deterministic restarts independent of global random seeds or PyTorch's
        initialization routines.
        """
        with torch.no_grad():
            self.lin_estimator.weight.copy_(self.W_init)
            self.lin_estimator.bias.copy_(self.b_init)


    def reset_parameters(self, rng : torch.Generator):
        r"""
        Reinitialize the model parameters.

        This resets the weights and bias of ``self.lin_estimator`` using
        PyTorch's default initialization for ``nn.Linear`` (Kaiming-uniform
        for weights and uniform bias based on fan-in). It copies the implementation
        of ``nn.Linear.reset_parameters`` while using a specific rng
        """
        # Weight resetting
        nn.init.kaiming_uniform_(self.lin_estimator.weight, a=sqrt(5), generator=rng)

        # Bias resetting
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin_estimator.weight)
        bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.lin_estimator.bias, -bound, bound, generator=rng)

    
    def fit(self, X : torch.Tensor, y : torch.Tensor, reduction : str = "sum"):
        r"""
        Fit the logistic regression model via full-batch L-BFGS.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``[..., n_features]``. All leading dimensions are
            flattened into a single batch for optimization (only works for binary 
            classification with different leading dimensions, in multiclass it 
            needs to have the shape ``[N, n_features]``)

        y : torch.Tensor
            Target tensor. 
            - **For binary classification:** must contain values ``0`` or ``1`` 
              and have shape ``[...]`` matching the leading dimensions of ``X``.
            - **For multinomial classification**, must contain integer class labels
              in ``{0, ..., n_classes-1}`` and have the shape [N,]

        reduction : Optional[str], default = "sum"
            ``reduction`` argument for the cross entropy loss. Either ``'sum'`` or  
            ``'mean'``.

        Returns
        -------
        self : TorchLogistic
            The fitted model.

        Warnings
        --------
        - L-BFGS performs **multiple internal iterations** inside a single call to
          ``optimizer.step``. This is not equivalent to a single gradient update.
        - All leading dimensions of ``X`` and ``y`` are treated as a single batch.
          There is no notion of per-group or per-sequence optimization.
        """
        if self.n_classes==2:
            loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            loss_fn = nn.CrossEntropyLoss(reduction=reduction)

        optimizer = optim.LBFGS(self.parameters(), **self.lbfgs_kwargs)

        def train_step():
            optimizer.zero_grad()
            logits = self(X)
            loss = loss_fn(logits, y)
            loss.backward()
            return loss
        
        self.train()
        optimizer.step(train_step)

        return self
    
    def to_state_dict(self) -> Dict[str, ndarray]:
        r"""
        Return a lightweight, device-agnostic snapshot of the model parameters.

        This method extracts the weight and bias of the internal linear estimator,
        moves them to CPU, and converts them to NumPy arrays. The resulting dictionary
        is fully pickle-safe and independent of the device on which the model currently
        resides.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A dictionary with keys ``"weight"`` and ``"bias"``, each containing a
            CPU-resident NumPy array representing the corresponding parameter.

        Notes
        -----
        - The returned arrays are detached copies; modifying them does not affect
        the model.
        - This method is intended for fast snapshotting in hot paths where full
        ``state_dict`` serialization would be unnecessarily heavy.
        """

        return {
            "weight" : self.lin_estimator.weight.detach().cpu().numpy(),
            "bias" : self.lin_estimator.bias.detach().cpu().numpy(),
            "lbfgs_kwargs" : self.lbfgs_kwargs.copy()
        }
    
    def load_from_state_dict(self, state_dict : Dict[str, Union[ndarray, torch.Tensor]]):
        r"""
        Load model parameters from a lightweight state dictionary.

        This method restores the weight and bias of the internal linear estimator
        from a dictionary produced by :meth:`copied_state_dict`. The input may contain
        either NumPy arrays or PyTorch tensors. All data is converted to the correct
        dtype and moved to the device of the existing parameters.

        Parameters
        ----------
        state_dict : Dict[str, Union[numpy.ndarray, torch.Tensor]]
            A dictionary containing exactly the keys ``"weight"`` and ``"bias"``.
            Each entry must be either a NumPy array or a PyTorch tensor with a shape
            matching the corresponding parameter.

        Raises
        ------
        ValueError
            If required keys are missing, if unexpected keys are present, or if the
            provided data is not a NumPy array or tensor.

        Notes
        -----
        - This method does not rely on PyTorch's ``load_state_dict`` and is intended
        for fast restoration of small models in performance-sensitive code paths.
        - Parameter shapes are expected to match exactly; no broadcasting or reshaping
        is performed.
        """
        if not ("bias" in state_dict and "weight" in state_dict):
            raise ValueError("state_dict does not contain bias and weight")
        if len(state_dict)>2:
            raise ValueError("state_dict contains more than only 'bias' and 'weight' als keys")
        
        with torch.no_grad():
            for param_name in ["bias", "weight"]:
                saved_param_data = state_dict.get(param_name)
                if isinstance(saved_param_data, ndarray):
                    saved_param_data = torch.from_numpy(saved_param_data)
                elif not isinstance(saved_param_data, torch.Tensor):
                    raise ValueError("Data type not recognized")
                
                param = getattr(self.lin_estimator, param_name)
                if saved_param_data.shape != param.shape:
                    raise ValueError(f"Shape mismatch for {param_name}: expected {param.shape}, got {saved_param_data.shape}")

                param.copy_(saved_param_data.to(param.dtype).to(param.device))

        if "lbfgs_kwargs" in state_dict:
            self.lbfgs_kwargs = deepcopy(state_dict["lbfgs_kwargs"])
    
    @classmethod
    def instantiate_from_state_dict(cls, state_dict : dict, device: torch.device = None):
        expected_keys = ["weight", "bias", "lbfgs_kwargs"]
        if set(state_dict.keys()) != set(expected_keys):
            keys_str_expected = ", ".join([f"'{k}'" for k in expected_keys])
            keys_str_found = ", ".join([f"'{k}'" for k in state_dict.keys()])
            raise KeyError(
                "state_dict has to contain exactly the keys " + keys_str_expected + ". Contains: " + keys_str_found
            )
        
        if not isinstance(state_dict["weight"], (ndarray, torch.Tensor)):
            raise TypeError("state_dict['weight'] has to contain an np.ndarray or a torch.Tensor")
        
        
        weight_data = state_dict["weight"]

        if weight_data.ndim != 2:
            raise AssertionError("state_dict['weight'] should be a 2d np.ndarray or torch.Tensor")
        n_classes, n_features = weight_data.shape
        n_classes = max(n_classes, 2)

        dtype = weight_data.dtype
        if isinstance(weight_data, ndarray):
            dtype=getattr(torch, str(dtype))
        
        instance = cls(n_features, n_classes, state_dict["lbfgs_kwargs"], device=device, dtype=dtype, secure_init=True)

        filtered_state_dict = {k : v for k,v in state_dict.items() if k!="lbfgs_kwargs"}
        instance.load_from_state_dict(filtered_state_dict)

        return instance

@contextmanager
def _cusolver_context(device: torch.device):
    """
    Temporarily set the preferred linalg library to cuSOLVER on CUDA devices.
    No-op on CPU. Restores the previous setting on exit, even if an exception
    is raised inside the block.
    """
    if device.type != "cuda":
        yield
        return
    prev = torch.backends.cuda.preferred_linalg_library()
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
        yield
    finally:
        torch.backends.cuda.preferred_linalg_library(prev)

class BatchedLogistic(nn.Module):
    r"""
    Batched binary logistic regression fitted via exact Newton-Raphson iterations,
    equivalent to Iteratively Re-weighted Least Squares (IRLS) for the logit link
    along each batch of observations.

    The Newton step at iteration :math:`t` for a single batch element is

    .. math::

        H_t &= \tilde{X}^{\top} W_t \tilde{X} + \lambda I, \quad
        W_t = \operatorname{diag}\!\bigl(\hat{p}_t(1-\hat{p}_t)\bigr), \\
        s_t &= \tilde{X}^{\top}(y - \hat{p}_t), \\
        \Delta\beta_t &= H_t^{-1} s_t
            \quad\text{(via Cholesky solve, never explicit inversion)},

    where :math:`\tilde{X}` is the design matrix, optionally augmented with a
    leading column of ones when ``fit_intercept=True``, and :math:`\lambda I` is a
    minimal Tikhonov regularisation term (``tikhonov = eps``) that acts as a
    numerical floor on the eigenvalues of :math:`H_t` without meaningfully biasing
    the MLE.  It is purely a numerical guard, not a statistical penalty.

    Convergence is declared independently per batch element once **both** criteria
    are satisfied:

    * :math:`\|s_t\|_\infty < \texttt{tol\_score}` — no score component is still
      meaningfully non-zero.
    * :math:`\|\Delta\beta_t\|_\infty < \texttt{tol\_step}` — parameters have
      stopped moving.  This guards against the ill-conditioned case where a small
      score norm can coexist with large parameter steps.

    Once a batch element converges it is **frozen**: its update is masked to zero
    for all subsequent iterations (early stopping per element).  Observations within
    each batch element can additionally be masked via ``mask_valid_obs``, allowing
    ragged or partially-observed designs without padding bias.

    Attributes
    ----------
    beta : torch.Tensor
        Coefficient buffer of shape ``[*batch_shape, p_aug, 1]``, where
        ``p_aug = n_features + 1`` when ``fit_intercept=True`` and ``n_features``
        otherwise.  When ``fit_intercept=True`` the first row of every coefficient
        vector corresponds to the intercept, consistent with the statistics
        convention of placing the constant in the first column of the design matrix.
    n_features : int
        Number of input features, excluding the implicit intercept if any.
    fit_intercept : bool
        Whether a leading intercept column is prepended to the design matrix.
    min_iter : int
        Minimum Newton iterations per :meth:`fit` call.
    max_iter : int
        Maximum Newton iterations per :meth:`fit` call.
    tol_score : float
        :math:`L^\infty` convergence tolerance on the score vector.
    tol_step : float
        :math:`L^\infty` convergence tolerance on the parameter update.

    Properties
    ----------
    batch_shape : torch.Size
        Leading dimensions of ``beta``, i.e. ``beta.shape[:-2]``.
    p_aug : int
        Total parameter dimension per batch element, i.e. ``beta.size(-2)``.
    device : torch.device
        Device on which ``beta`` resides.

    Methods
    -------
    fit(X, y, mask_valid_obs=None)
        Fit all batch elements simultaneously via batched Newton-Raphson.
    forward(X, betas_reshaped=None, output_shape_X=None)
        Compute raw logits for all batch elements.
    predict_proba(X)
        Return ``[p(y=0), p(y=1)]`` probabilities for all batch elements.
    predict(X)
        Return binary class predictions for all batch elements.
    decision_function(X)
        Alias for :meth:`forward`.
    reset_beta()
        Re-initialise ``beta`` to zero in-place.
    assert_inputs(X, y, mask_valid_obs)
        Validate shapes and device consistency of inputs.
    handle_shapes(X, y, mask_valid_obs)
        Expand and flatten leading dimensions for ``bmm``-compatible layout.
    to_state_dict()
        Lightweight, device-agnostic parameter snapshot.
    load_from_state_dict(state_dict)
        Restore parameters from a snapshot produced by :meth:`to_state_dict`.
    instantiate_from_state_dict(state_dict, device=None)
        Class method: construct a new instance from a snapshot.
    from_torch_logistic(torch_logistic, batch_shape, ...)
        Static method: construct a :class:`BatchedLogistic` from a
        :class:`TorchLogistic` instance, inheriting its structural configuration.

    Notes
    -----
    **Why** ``bmm`` **throughout?**
    Every matrix product in :meth:`fit` and :meth:`forward` uses ``torch.bmm``
    over the explicitly flattened batch dimension so that each batch element's
    computation is provably independent.  There is no broadcasting across batch
    elements — ``beta`` has shape ``[*batch_shape, p_aug, 1]`` and is updated
    element-wise via in-place ``add_``.

    **Why Cholesky instead of explicit inversion?**
    For any SPD matrix :math:`H`, forming :math:`H^{-1}` explicitly squares the
    condition number :math:`\kappa(H)`.  Cholesky factorisation followed by
    triangular solves propagates only :math:`\kappa(H)`, which is critical for
    folds where limited data or near-multicollinearity makes :math:`H` borderline.
    This is the same principle as computing the Mahalanobis distance via a
    triangular solve rather than an explicit inverse of the covariance matrix.

    **Why zero initialisation?**
    Unlike first-order methods (SGD, Adam), Newton's convergence rate is
    independent of initialisation once inside the quadratic convergence basin —
    which for strictly concave objectives like the logistic log-likelihood is
    essentially the entire parameter space.  He/Xavier initialisation buys nothing
    here because Newton self-corrects via curvature from the very first step.

    **Statistical interpretation**
    Because this implements exact Newton on the log-likelihood, the fixed point is
    the MLE.  The Cholesky factor :math:`L` of the final :math:`H` satisfies
    :math:`\operatorname{Var}(\hat\beta) = H^{-1}`, so asymptotic standard errors
    are available from :math:`L` via ``torch.cholesky_inverse`` without ever
    forming :math:`H^{-1}` explicitly.
    """

    def __init__(
        self,
        n_features: int,
        batch_shape: Union[torch.Size, Tuple[int, ...]],
        fit_intercept: bool = True,
        min_iter: int = 3,
        max_iter: int = 100,
        tol_score: Optional[float] = None,
        tol_step: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        init_betas: Optional[torch.Tensor] = None,
    ):
        r"""
        Initialise the batched logistic regression model.

        Parameters
        ----------
        n_features : int
            Number of input features, excluding the implicit intercept if any.
        batch_shape : torch.Size or tuple of int
            Shape of the leading batch dimensions, e.g. ``(k,)`` for k-fold CV or
            ``(B1, B2, k)`` for a nested grid.  Each element of the batch maintains
            a fully independent coefficient vector.
        fit_intercept : bool, default=True
            If ``True``, a column of ones is prepended to the design matrix so that
            ``beta`` contains an intercept term.  Mirrors the ``fit_intercept``
            convention of scikit-learn and the ``bias`` convention of
            ``torch.nn.Linear``.
        min_iter : int, default=3
            Minimum number of Newton iterations performed regardless of convergence.
            Prevents premature termination near the zero initialisation.
        max_iter : int, default=100
            Maximum number of Newton iterations per :meth:`fit` call.
        tol_score : float, optional
            Convergence tolerance for the :math:`L^\infty` score norm.  Defaults to
            ``eps ** (2/3)`` of the working dtype, mirroring ``tolerance_grad`` in
            :class:`TorchLogistic`.
        tol_step : float, optional
            Convergence tolerance for the :math:`L^\infty` step norm.  Defaults to
            ``eps ** (7/8)`` of the working dtype, mirroring ``tolerance_change`` in
            :class:`TorchLogistic`.
        device : torch.device, optional
            Device on which all tensors are allocated.
        dtype : torch.dtype, optional
            Floating-point dtype.  Defaults to ``torch.get_default_dtype()``.
        init_betas : torch.Tensor, optional
            Initial value for ``beta``.  If provided, must have shape
            ``[*batch_shape, p_aug, 1]`` or ``[*batch_shape, p_aug]`` (a trailing
            singleton is added automatically in the latter case), where
            ``p_aug = n_features + 1`` if ``fit_intercept`` else ``n_features``.
            If ``None``, ``beta`` is initialised to zero, which is the canonical
            and recommended choice for Newton-Raphson on a strictly concave
            objective.
        """
        super().__init__()

        self.n_features    = int(n_features)
        self.fit_intercept = bool(fit_intercept)
        self.min_iter      = int(min_iter)
        self.max_iter      = int(max_iter)

        _dtype = dtype if dtype is not None else torch.get_default_dtype()
        _eps   = torch.finfo(_dtype).eps

        self.tol_score = float(tol_score) if tol_score is not None else float(_eps ** (2 / 3))
        self.tol_step  = float(tol_step)  if tol_step  is not None else float(_eps ** (7 / 8))
        self._tikhonov = float(_eps)

        # p_aug: effective parameter count per batch element.
        # With fit_intercept the design matrix is [1 | X] (intercept first,
        # following the statistics convention), so beta[..., 0, :] is the intercept.
        p_aug      = self.n_features
        if fit_intercept:
            p_aug += 1
        beta_shape = torch.Size(batch_shape) + (p_aug, 1)

        if init_betas is None:
            init_betas = torch.zeros(beta_shape, dtype=_dtype, device=device)
        else:
            if init_betas.shape == beta_shape[:-1]:
                init_betas = init_betas.unsqueeze(-1)
            elif init_betas.shape != beta_shape:
                raise AssertionError(
                    "init_betas must have shape [*batch_shape, p_aug, 1] or "
                    "[*batch_shape, p_aug], where p_aug = n_features (+ 1 if "
                    f"fit_intercept).  Got {tuple(init_betas.shape)}."
                )

        self.register_buffer("beta", init_betas.clone().to(dtype=_dtype, device=device))

        # _augment is resolved once at init to avoid branching at call time,
        # mirroring the logit_to_probs pattern in TorchLogistic.
        if self.fit_intercept:
            self._augment: Callable[[torch.Tensor], torch.Tensor] = self._prepend_ones
        else:
            self._augment = nn.Identity()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepend_ones(X: torch.Tensor) -> torch.Tensor:
        r"""
        Prepend a column of ones to ``X`` to absorb the intercept into ``beta``.

        Placing the constant in the **first** column follows the statistics
        convention (e.g. R's model matrices, classical GLM textbooks), so
        ``beta[..., 0, :]`` always corresponds to the intercept term.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]``.

        Returns
        -------
        torch.Tensor
            Shape ``[*batch_shape, N, n_features + 1]`` with a leading ones column.
        """
        return torch.nn.functional.pad(X, pad=(1,0), mode='constant', value=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        r"""
        Device on which ``beta`` resides.

        Returns
        -------
        torch.device
        """
        return self.beta.device

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Leading batch dimensions of ``beta``, i.e. ``beta.shape[:-2]``.

        Returns
        -------
        torch.Size
        """
        return self.beta.shape[:-2]

    @property
    def p_aug(self) -> int:
        r"""
        Total parameter dimension per batch element, i.e. ``beta.size(-2)``.

        Equal to ``n_features + 1`` when ``fit_intercept=True``, otherwise
        ``n_features``.

        Returns
        -------
        int
        """
        return self.beta.size(-2)

    # ------------------------------------------------------------------
    # Input validation and shape utilities
    # ------------------------------------------------------------------

    def assert_inputs(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask_valid_obs: Optional[torch.Tensor] = None,
        for_training: bool = False
    ) -> None:
        r"""
        Validate shapes and device consistency of inputs.

        Parameters
        ----------
        X : torch.Tensor
            Design matrix.  Must have at least ``self.beta.dim()`` dimensions,
            with ``X.size(-1) == self.n_features`` and middle dimensions
            broadcastable against ``self.batch_shape``.
        y : torch.Tensor, optional
            Binary targets.  Must satisfy ``y.shape == X.shape[:-1]``.
        mask_valid_obs : torch.Tensor, optional
            Boolean validity mask.  Must satisfy
            ``mask_valid_obs.shape == X.shape[:-1]``.

        Raises
        ------
        AssertionError
            On any shape inconsistency or device mismatch.
        """
        if y is not None:
            assert_tensors(X, y, tensor_names="X, y",
                           checks=["are_tensors", "same_device"])
            if X.shape[:-1] != y.shape:
                raise AssertionError("X and y do not have the right dimensions")

        if mask_valid_obs is not None:
            assert_tensors(X, mask_valid_obs, tensor_names="X, mask_valid_obs",
                           checks=["are_tensors", "same_device"])
            if X.shape[:-1] != mask_valid_obs.shape:
                raise AssertionError(
                    "X and mask_valid_obs do not have the right dimensions"
                )

        assert_tensors(X, self.beta, tensor_names="X, beta",
                       checks=["same_dtype", "same_device"])
        
        if X.size(-1) != self.n_features:
            raise AssertionError(
                f"X has {X.size(-1)} features but model expects {self.n_features}"
            )
        
        if for_training:
            if X.shape[:-2] != self.beta.shape[:-2]:
                raise AssertionError(f"X must have a shape for training with same batch dimensions as beta, i.e. self.beta.shape[:-2]={self.beta.shape[:-2]}")
            
            return
        
        try:
            torch.broadcast_shapes(X.shape[:-2], self.beta.shape[:-2])
        except RuntimeError:
            raise AssertionError("The leading dimension of X and self.beta are not broadcastable")

    # ------------------------------------------------------------------
    # Forward / prediction
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute raw logits for all batch elements.

        Can be called in two modes:

        * **Standard mode** (``betas_reshaped=None, output_shape_X=None``):
          :meth:`handle_shapes` is invoked internally.  Suitable for inference.
        * **Pre-shaped mode**: caller supplies the already-flattened ``betas``
          and the target output shape, skipping the internal :meth:`handle_shapes`
          call.  Useful when :meth:`handle_shapes` has already been called in an
          outer loop (e.g. inside :meth:`fit`).  Both arguments must be provided
          together; supplying only one raises ``RuntimeError``.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]`` in standard mode, or
            ``[B, N, n_features]`` in pre-shaped mode.
        betas_reshaped : torch.Tensor, optional
            Flattened betas of shape ``[B, p_aug, 1]``.  Must be provided
            together with ``output_shape_X``.
        output_shape_X : torch.Size, optional
            Full pre-flatten shape of ``X``, used to restore the leading
            dimensions of the output.  Must be provided together with
            ``betas_reshaped``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[*batch_shape, N]``.

        Raises
        ------
        RuntimeError
            If exactly one of ``betas_reshaped`` / ``output_shape_X`` is provided.
        """

        X_aug = self._augment(X)                      # [*batch_shape, N, p_aug]
        return  X_aug @ self.beta                     # [*batch_shape, N, 1]
    
    def predict_prob_bad(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(X))   # [*batch_shape, N, 1]

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute binary class probabilities for all batch elements.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``[*batch_shape, N, 2]`` with columns
            ``[p(y=0), p(y=1)]``, consistent with :class:`TorchLogistic`.
        """
        p1 = self.predict_prob_bad(X)   # [*batch_shape, N, 1]
        return torch.cat([1.0 - p1, p1], dim=-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Return binary class predictions for all batch elements.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]``.

        Returns
        -------
        torch.Tensor
            Integer labels of shape ``[*batch_shape, N]``.
        """
        return torch.argmax(self.predict_proba(X), dim=-1)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute raw decision scores (logits).

        Alias for :meth:`forward` provided for consistency with
        :class:`TorchLogistic` and scikit-learn conventions.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[*batch_shape, N]``.
        """
        return self.forward(X)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def reset_beta(self) -> None:
        r"""
        Re-initialise ``beta`` to zero in-place.

        Notes
        -----
        Zero initialisation is appropriate for Newton-Raphson on a strictly
        concave log-likelihood: the curvature landscape guarantees convergence
        from any starting point, and zero is the canonical uninformative start.
        Unlike first-order methods, Newton self-corrects via curvature from the
        very first step, so He/Xavier initialisation would provide no benefit.
        The in-place ``zero_()`` preserves the buffer registration in the module.
        """
        with torch.no_grad():
            self.beta.zero_()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        mask_valid_obs: Optional[torch.Tensor] = None,
        prefer_cusolver_when_on_cuda: bool = True
    ) -> "BatchedLogistic":
        r"""
        Fit all batch elements simultaneously via batched exact Newton-Raphson.

        Parameters
        ----------
        X : torch.Tensor
            Design matrix of shape ``[*batch_shape, N, n_features]``.  Each
            leading batch element receives its own independent design matrix.
        y : torch.Tensor
            Binary targets of shape ``[*batch_shape, N]`` with values in
            ``{0, 1}``.
        mask_valid_obs : torch.Tensor, optional
            Boolean tensor of shape ``[*batch_shape, N]``.  ``True`` marks
            observations that are valid and should contribute to the likelihood.
            ``False`` marks missing or invalid observations: their corresponding
            rows of the augmented design matrix are zeroed out before the Newton
            loop so they contribute nothing to the score or Hessian.  Batch
            elements where **no** observation is valid
            (``mask_valid_obs[b].any() == False``) are excluded from fitting
            entirely and their ``beta`` is left at zero.  If ``None``, all
            observations in all batch elements are treated as valid.

        Returns
        -------
        self : BatchedLogistic
            The fitted model.  ``self.beta`` is updated in-place for all active
            batch elements; elements with no valid observations are untouched.

        Notes
        -----
        **Why perfer cusolver backend?**
        If the used device is cuda then the whole method will be executed
        on the cuSOLVER backend. The reason is that ``torch.cholesky_solve``
        gets passed to MAGMA for batches larger than one which for some
        machines is not supported.


        **Observation masking**

        Zeroing the rows of :math:`\tilde{X}` corresponding to invalid
        observations is numerically equivalent to removing those observations:
        since :math:`X_{\text{zero}}^{\top} r = 0` for any residual vector
        :math:`r`, masked rows contribute nothing to :math:`s_t` or :math:`H_t`.
        The corresponding ``y`` entries are set to ``-1`` (a dummy value); their
        residuals are structurally zeroed by the zeroed ``X`` rows and never
        enter the score.

        **Per-element independence**

        All matrix products use ``torch.bmm`` over the explicitly flattened
        batch dimension ``B = prod(batch_shape)`` so that each batch element's
        computation is provably independent.

        **Newton step**

        .. math::

            H_t &= \tilde{X}^{\top} W_t \tilde{X} + \lambda I, \\
            s_t &= \tilde{X}^{\top}(y - \hat{p}_t), \\
            \Delta\beta_t &= H_t^{-1} s_t
                \quad\text{(Cholesky solve)},

        where :math:`\lambda = \varepsilon_{\text{machine}}` is the Tikhonov
        guard.

        **Convergence and early stopping**

        A batch element is declared converged once both tolerance criteria are
        met after at least ``min_iter`` iterations.  Frozen elements receive a
        zero update via multiplication by the ``active`` float mask, so ``beta``
        is never modified after convergence.  Multiplying by the mask rather than
        using ``masked_fill_`` avoids in-place mutation of the ``cholesky_solve``
        output buffer.
        """
        self.assert_inputs(X, y, mask_valid_obs, for_training = True)

        with _cusolver_context(self.device), torch.no_grad():
            self.reset_beta()
            # N     = X.size(-2)
            p_aug = self.p_aug # float
            beta  = self.beta                       # [*batch_dims, p_aug, 1]
            y     = y.to(beta.dtype).unsqueeze(-1) # [*batch_dims, N, 1]
            X_aug = self._augment(X)               # [*batch_dims, N, p_aug]
            batch_dims = beta.shape[:-2]
            if len(batch_dims) == 0:
                y, X_aug, beta = [t.unsqueeze(0) for t in (y, X_aug, beta)]
                batch_dims = torch.Size([1])

            if mask_valid_obs is not None:
                # Zero invalid rows of X_aug — masked rows then contribute nothing
                # to X^T r or X^T W X regardless of logit or residual values.
                # Set corresponding y to a dummy; structurally zeroed by X_aug.
                X_aug.masked_fill_(~mask_valid_obs.unsqueeze(-1), 0.0)
                y.masked_fill_(~mask_valid_obs.unsqueeze(-1), -1.0)

            # Tikhonov guard: eps * I, built once per fit call.
            # repeat (not expand) ensures contiguous memory for bmm.
            tikhonov = (
                self._tikhonov
                * torch.eye(p_aug, dtype=beta.dtype, device=beta.device)
            )                                                          # [p_aug, p_aug]

            active = beta.new_ones(batch_dims, dtype=bool)

            for iteration in range(self.max_iter):
                # ---- forward pass ----------------------------------------
                logits = X_aug @ beta                             # [*batch_dims, N, 1]
                p_hat  = torch.sigmoid(logits)                    # [*batch_dims, N, 1]

                # ---- IRLS weights ----------------------------------------
                W = p_hat * (1.0 - p_hat)                         # [*batch_dims, N, 1]

                # ---- score  s = X^T (y - p_hat) --------------------------
                residual = y - p_hat                              # [*batch_dims, N, 1]
                score    = X_aug.mT @ residual                    # [*batch_dims, p_aug, 1]

                # ---- Hessian  H = X^T W X + lambda I --------------------
                # Weighting rows of X by W then using bmm avoids forming the
                # dense [B, N, N] diagonal weight matrix explicitly — equivalent
                # to diag(W) @ X but O(N * p) instead of O(N^2 + N * p).
                X_w = X_aug * W                                    # [*batch_dims, N, p_aug]
                H   = (X_aug.mT @ X_w) + tikhonov                  # [*batch_dims, p_aug, p_aug]

                # ---- Newton step via Cholesky solve ----------------------
                # H is SPD by construction (Tikhonov guard ensures this even
                # under separation or multicollinearity).
                # cholesky_solve avoids squaring kappa(H) vs. explicit inversion.
                L          = torch.linalg.cholesky(H)             # [*batch_dims, p_aug, p_aug]
                delta_beta = torch.cholesky_solve(score, L)       # [*batch_dims, p_aug, 1]

                # ---- convergence check (L-inf on score AND step) ---------
                score_linf = score.abs().amax(dim=(-2, -1))        # [*batch_dims]
                step_linf  = delta_beta.abs().amax(dim=(-2, -1))   # [*batch_dims]

                newly_converged = (
                    (score_linf < self.tol_score) &
                    (step_linf  < self.tol_step)
                )

                if iteration >= self.min_iter:
                    active &= ~newly_converged

                if not active.any():
                    break

                # ---- masked in-place update ------------------------------
                # Multiply by float active mask rather than masked_fill_ to
                # avoid in-place mutation of the cholesky_solve output buffer.
                beta.add_(delta_beta * active.view(*batch_dims, 1, 1))

            # Write fitted betas back into the registered buffer.
            # batch_mask indexes the flattened batch dimension; reshaping to
            # batch_shape recovers the correct multi-dimensional boolean index.
            if beta.untyped_storage().data_ptr() != self.beta.untyped_storage().data_ptr():
                self.beta = beta

            return self

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def to_state_dict(self) -> dict:
        r"""
        Return a lightweight, device-agnostic snapshot of the model state.

        Returns
        -------
        dict
            Dictionary with the following keys:

            ``"beta"``
                ``numpy.ndarray`` of shape ``[*batch_shape, p_aug, 1]``.
            ``"n_features"``
                ``int``.
            ``"batch_shape"``
                ``tuple`` of ``int``.
            ``"fit_intercept"``
                ``bool``.
            ``"min_iter"``
                ``int``.
            ``"max_iter"``
                ``int``.
            ``"tol_score"``
                ``float``.
            ``"tol_step"``
                ``float``.

        Notes
        -----
        The returned arrays are detached CPU copies; modifying them does not
        affect the model.  The dictionary is fully pickle-safe and
        device-independent, consistent with :meth:`TorchLogistic.to_state_dict`.
        """
        return {
            "beta":          self.beta.detach().cpu().numpy(),
            "n_features":    self.n_features,
            "batch_shape":   tuple(self.batch_shape),
            "fit_intercept": self.fit_intercept,
            "min_iter":      self.min_iter,
            "max_iter":      self.max_iter,
            "tol_score":     self.tol_score,
            "tol_step":      self.tol_step,
        }

    def load_from_state_dict(self, state_dict: dict) -> None:
        r"""
        Restore model state from a snapshot produced by :meth:`to_state_dict`.

        Parameters
        ----------
        state_dict : dict
            Must contain exactly the keys produced by :meth:`to_state_dict`.

        Raises
        ------
        ValueError
            If required keys are missing, unexpected keys are present, shapes do
            not match, or ``"beta"`` is neither a NumPy array nor a PyTorch tensor.
        """
        expected = {
            "beta", "n_features", "batch_shape", "fit_intercept",
            "min_iter", "max_iter", "tol_score", "tol_step",
        }
        given = set(state_dict.keys())
        if given != expected:
            raise ValueError(
                f"state_dict keys mismatch.\n  Expected: {sorted(expected)}"
                f"\n  Got:      {sorted(given)}"
            )

        beta_data = state_dict["beta"]
        if isinstance(beta_data, ndarray):
            beta_data = torch.from_numpy(beta_data)
        elif not isinstance(beta_data, torch.Tensor):
            raise ValueError(
                f"'beta' must be a numpy.ndarray or torch.Tensor, "
                f"got {type(beta_data)}"
            )
        if beta_data.shape != self.beta.shape:
            raise ValueError(
                f"Shape mismatch for 'beta': "
                f"expected {tuple(self.beta.shape)}, got {tuple(beta_data.shape)}"
            )

        with torch.no_grad():
            self.beta.copy_(
                beta_data.to(dtype=self.beta.dtype, device=self.beta.device)
            )

        self.min_iter      = int(state_dict["min_iter"])
        self.max_iter      = int(state_dict["max_iter"])
        self.tol_score     = float(state_dict["tol_score"])
        self.tol_step      = float(state_dict["tol_step"])
        self.fit_intercept = bool(state_dict["fit_intercept"])

        # Re-bind _augment in case fit_intercept changed relative to init.
        if self.fit_intercept:
            self._augment = self._prepend_ones
        else:
            self._augment = lambda X: X

    @classmethod
    def instantiate_from_state_dict(
        cls,
        state_dict: dict,
        device: Optional[torch.device] = None,
    ) -> "BatchedLogistic":
        r"""
        Construct a new :class:`BatchedLogistic` instance from a snapshot.

        Parameters
        ----------
        state_dict : dict
            Dictionary as produced by :meth:`to_state_dict`.
        device : torch.device, optional
            Target device for the new instance.  Defaults to CPU.

        Returns
        -------
        BatchedLogistic
            A fully initialised instance with parameters restored from
            ``state_dict``.

        Raises
        ------
        ValueError
            If ``state_dict`` keys or tensor shapes are inconsistent.
        """
        expected = {
            "beta", "n_features", "batch_shape", "fit_intercept",
            "min_iter", "max_iter", "tol_score", "tol_step",
        }
        given = set(state_dict.keys())
        if given != expected:
            raise ValueError(
                f"state_dict keys mismatch.\n  Expected: {sorted(expected)}"
                f"\n  Got:      {sorted(given)}"
            )

        beta_data = state_dict["beta"]
        if isinstance(beta_data, ndarray):
            beta_data = torch.from_numpy(beta_data)
        elif not isinstance(beta_data, torch.Tensor):
            raise ValueError(
                f"'beta' must be a numpy.ndarray or torch.Tensor, "
                f"got {type(beta_data)}"
            )

        instance = cls(
            n_features    = int(state_dict["n_features"]),
            batch_shape   = tuple(state_dict["batch_shape"]),
            fit_intercept = bool(state_dict["fit_intercept"]),
            min_iter      = int(state_dict["min_iter"]),
            max_iter      = int(state_dict["max_iter"]),
            tol_score     = float(state_dict["tol_score"]),
            tol_step      = float(state_dict["tol_step"]),
            device        = device,
            dtype         = beta_data.dtype,
        )
        with torch.no_grad():
            instance.beta.copy_(
                beta_data.to(dtype=beta_data.dtype, device=device)
            )
        return instance

class PerfectBayesClassifier:
    r"""
    Perfect Bayesian classifier for binary classification derived from a known DGP.

    Combines two Gaussian mixture components (bad and good outcomes) with optional
    idiosyncratic shocks to compute the optimal Bayesian classification rule.
    Predictions are based on the posterior probability of the bad outcome given features,
    computed via the normalized likelihood ratio of the mixture components.

    The classifier is parameterized by the conditional distributions :math:`p(X | Y=b)`
    and :math:`p(X | Y=g)` modeled as Gaussian mixtures, along with the prior
    probability :math:`\mathbb{P}(Y=b)`. When idiosyncratic shocks are enabled,
    the posterior incorporates an additional unconditional shock probability that
    can flip the outcome independently of features.

    Attributes
    ----------
    prob_bad_given_no_shock : float
        Prior probability of bad outcome, :math:`\mathbb{P}(Y=b \mid \text{no shock})`.
        Derived from ``dgp.prob_bad_given_no_shock``.
    prob_shock : Optional[float]
        Probability of idiosyncratic shock, :math:`\mathbb{P}(\text{shock})``.
        ``None`` if ``consider_idiosyncratic_shock=False``.
    prob_bad_because_of_shock : float
        Unconditional probability of bad outcome due to shock alone:
        :math:`\mathbb{P}(\text{shock}) \times \mathbb{P}(Y=b \mid \text{shock})`.
        Only set if ``consider_idiosyncratic_shock=True``.
    joint_means : torch.Tensor
        Concatenated means of bad and good mixture components. Shape
        ``[*batch_dims, K, F]`` where ``K = K_bad + K_good``.
    joint_cov_chols : torch.Tensor
        Concatenated Cholesky factors of covariance matrices for bad and good
        components. Shape ``[*batch_dims, K, F, F]``.
    joint_inv_covs: torch.Tensor
        Concatenated inverse of the covariance matrices calculated via
        ``torch.cholesky_inverse``. Used for performance reasons for the
        mahalanobis standardization.
    joint_weights : torch.Tensor
        Concatenated mixture weights (scaled by class priors) for bad and good
        components. Shape ``[*batch_dims, K]``.

    Properties
    ----------
    dtype : torch.dtype
        Data type of ``joint_means`` (propagates to all computations).
    K : int
        Total number of mixture components (``K_bad + K_good``).
    K_bad : int
        Number of components in bad outcome mixture.
    K_good : int
        Number of components in good outcome mixture.
    F : int
        Feature dimension (number of features).
    batch_dims : torch.Size
        Leading batch dimensions of the mixture parameters.
    consider_idiosyncratic_shock : bool
        Whether idiosyncratic shocks are enabled (equivalently, ``prob_shock is not None``).

    Methods
    -------
    broadcast_X(X, safety_checks=True)
        Reshape feature matrix to align with batch dimensions for mixture evaluation.
    mahalanobis_X(X_broadcasted)
        Compute squared Mahalanobis distances from each feature vector to each
        mixture component.
    log_component_pdf(X_broadcasted)
        Compute log-probability density under each mixture component.
    log_prob_bad_given_no_shock(X_broadcasted)
        Compute log posterior of bad outcome (conditioned on no shock).
    predict_prob_bad(X, safety_checks=True)
        Compute posterior probability of bad outcome, optionally incorporating
        idiosyncratic shocks.
    predict_proba(features)
        Return class probabilities in the format ``[..., 2]`` where columns are
        ``[P(Y=0), P(Y=1)]`` (good, bad).
    extract_mixture_comps(mixture, prior_prob_given_no_shock, consider_feats_noise, feats_noise_std)
        Static method: extract and optionally augment means, Cholesky factors,
        and weights from a Gaussian mixture.
    log_normalizing_factors(cov_chol, weights)
        Static method: compute the log-space normalization constant for each
        mixture component (excluding the Mahalanobis term).

    Notes
    -----
    **Posterior computation via log-space**

    The posterior :math:`\mathbb{P}(Y=b \mid X)` is computed via the log-space
    likelihood ratio to avoid numerical underflow:

    .. math::

        \log \mathbb{P}(Y=b \mid X) = \log \Phi_b(X) - \log \Phi(X),

    where :math:`\Phi_b(X) = \sum_{k=1}^{K_b} p_k^{(b)} \phi_{k,b}(X)` is the
    marginal likelihood under the bad mixture (with :math:`\phi_{k,b}` being
    the Gaussian density of component :math:`k`), and :math:`\Phi(X)` is the
    total marginal likelihood across both mixtures.  Both are computed via
    ``torch.logsumexp`` on the per-component log-densities.

    **Idiosyncratic shocks**

    When enabled, the final predicted probability incorporates an unconditional
    shock term:

    .. math::

        \mathbb{P}(Y=b \mid X) = (1 - p_s) \cdot \mathbb{P}(Y=b \mid X, \text{no shock})
                                + p_s \cdot \mathbb{P}(Y=b \mid \text{shock}),

    where :math:`p_s` is the shock probability.  This models a background failure
    mode independent of the observed features.

    **Batch support**

    The classifier can handle batched mixture parameters (e.g., per-fold or
    per-scenario Gaussian mixtures).  All feature evaluations preserve batch
    dimensions through explicit reshaping and broadcasting.
    """

    # Instance variable type hints
    prob_bad_given_no_shock: float
    prob_shock: Optional[float]
    prob_bad_because_of_shock: Optional[float]
    joint_means: torch.Tensor
    joint_cov_chols: torch.Tensor
    joint_inv_cov: torch.Tensor
    joint_weights: torch.Tensor

    def __init__(
            self,
            dgp: CreditDataGenerator,
            consider_idiosyncratic_shock: bool,
            consider_feats_noise: bool,
            var_to_hide: Optional[int] = None
        ):
        r"""
        Initialize the perfect Bayesian classifier from a data generating process.

        Parameters
        ----------
        dgp : CreditDataGenerator
            Data generating process instance providing the mixture parameters
            for bad and good outcomes, along with shock probabilities and
            prior probabilities.
        consider_idiosyncratic_shock : bool
            If ``True``, the posterior incorporates an unconditional shock term
            that can flip the outcome independently of features.  If ``False``,
            predictions are based entirely on the likelihood ratio.
        consider_feats_noise : bool
            If ``True``, augments the Gaussian mixture covariances with
            feature noise (typically observation error) from ``dgp.feats_noise_std``.

        Notes
        -----
        The classifier extracts and concatenates the bad and good mixture
        components, scaling their weights by the respective class priors.
        All tensors are stored on the device of the DGP's mixture tensors.
        """
        self.prob_bad_given_no_shock = dgp.prob_bad_given_no_shock
        if consider_idiosyncratic_shock:
            self.prob_shock = dgp.prob_idiosyncratic_shock
            self.prob_bad_because_of_shock = self.prob_shock * dgp.prob_bad_given_shock
        else:
            self.prob_shock = None
            self.prob_bad_because_of_shock = None

        self._is_batched = dgp.is_batched
        self._K_bad = dgp.bad_mixture.K

        # batch_dims = [] if not dgp.is_batched else [dgp.B]

        # [*batch_dims, K_bad, F], [*batch_dims, K_bad, F, F], [*batch_dims, K_bad]
        mean_bad, cov_chol_bad, weights_bad = self.extract_mixture_comps(
            dgp.bad_mixture,
            prior_prob_given_no_shock=self.prob_bad_given_no_shock,
            consider_feats_noise=consider_feats_noise,
            feats_noise_std= dgp.feats_noise_std,
            var_to_hide=var_to_hide
        )
        # [*batch_dims, K_good, F], [*batch_dims, K_good, F, F], [*batch_dims, K_good]
        mean_good, cov_chol_good, weights_good = self.extract_mixture_comps(
            dgp.good_mixture,
            prior_prob_given_no_shock=1-self.prob_bad_given_no_shock,
            consider_feats_noise=consider_feats_noise,
            feats_noise_std= dgp.feats_noise_std,
            var_to_hide=var_to_hide
        )

        self.joint_means = torch.cat([mean_bad, mean_good], dim=-2)             # [*batch_dims, K, F]
        self.joint_cov_chols = torch.cat([cov_chol_bad, cov_chol_good], dim=-3) # [*batch_dims, K, F, F]
        self.joint_inv_covs = torch.cholesky_inverse(
            self.joint_cov_chols.double(), # Use double precision for the inverse, as it is the most brittle step
            upper = False
        ).to(self.joint_cov_chols.dtype) # Turn back to the "usual precision"
        self.joint_weights = torch.cat([weights_bad, weights_good], dim=-1)     # [*batch_dims, K]

        self._log_component_const = self.log_normalizing_factors(self.joint_cov_chols, self.joint_weights) # [*batch_dims, K]

    @property
    def dtype(self) -> torch.dtype:
        r"""Data type of the mixture parameters.

        Returns
        -------
        torch.dtype
        """
        return self.joint_means.dtype

    @property
    def K(self) -> int:
        r"""Total number of mixture components.

        Equal to ``K_bad + K_good``.

        Returns
        -------
        int
        """
        return self.joint_weights.size(-1)

    @property
    def K_bad(self) -> int:
        r"""Number of components in the bad outcome mixture.

        Returns
        -------
        int
        """
        return self._K_bad
    
    @property
    def K_good(self) -> int:
        r"""Number of components in the good outcome mixture.

        Returns
        -------
        int
        """
        return self.K - self.K_bad
    
    @property
    def F(self) -> int:
        r"""Feature dimension (number of input features).

        Returns
        -------
        int
        """
        return self.joint_means.size(-1)
    
    @property
    def batch_dims(self) -> torch.Size:
        r"""Leading batch dimensions of the mixture parameters.

        Returns
        -------
        torch.Size
        """
        return self.joint_means.shape[:-2]

    @property
    def consider_idiosyncratic_shock(self) -> bool:
        r"""Whether idiosyncratic shocks are enabled.

        Equivalent to ``prob_shock is not None``.

        Returns
        -------
        bool
        """
        return self.prob_shock is not None
    
    @property
    def joint_covs(self) -> torch.Tensor:
        """
        Cvariance matrices of the bad and
        good components, the back transform of
        the cholesky decompositions

        Returns
        -------
        torch.Tensor
        """
        return self.joint_cov_chols @ self.joint_cov_chols.mT

    @staticmethod
    def extract_mixture_comps(
        mixture: GaussianMixture,
        prior_prob_given_no_shock: float,
        consider_feats_noise: bool,
        feats_noise_std: float,
        var_to_hide: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Extract and optionally augment components from a Gaussian mixture.

        Retrieves the normalized parameters (means, Cholesky factors, weights)
        from a mixture object. Optionally augments covariances with feature noise
        and scales weights by the class prior probability.

        Parameters
        ----------
        mixture : GaussianMixture
            Source mixture object with ``_normalized_params()`` method.
        prior_prob_given_no_shock : float
            Class prior probability, used to scale mixture weights.
        consider_feats_noise : bool
            If ``True``, augment covariances with feature noise.
        feats_noise_std : float
            Standard deviation of feature noise. Used only if
            ``consider_feats_noise=True``.

        Returns
        -------
        mean : torch.Tensor
            Component means, shape ``[K, F]`` or ``[B, K, F]`` depending on
            whether the mixture is batched.
        cov_chol : torch.Tensor
            Cholesky factors of covariances, shape ``[K, F, F]`` or
            ``[B, K, F, F]``.
        weights : torch.Tensor
            Mixture weights scaled by class prior, shape ``[K]`` or ``[B, K]``.
        """
        mean, cov_chol, weights = mixture._normalized_params() # [B, K, F], [B, K, F, F], [B, K]
        if var_to_hide is not None:
            F = mean.size(-1)
            if not (-F <= var_to_hide < F):
                raise AssertionError("var_to_hide outside of limit")
            model_vars = [f for f in range(F) if f != var_to_hide]
            mean = mean[..., model_vars]

            cov = cov_chol @ cov_chol.mT
            cov = cov[..., model_vars, :][..., :, model_vars]
            cov_chol = torch.linalg.cholesky(cov)
        if consider_feats_noise and feats_noise_std > 0:
            cov = mixture._effective_cov_additive(cov_chol @ cov_chol.mT, noise_var=feats_noise_std**2)
            cov_chol = torch.linalg.cholesky(cov)

        if weights is None:
            weights = mean.new_full(mean.shape[:-1], fill_value=prior_prob_given_no_shock) # [B, 1], K = 1
        else:
            weights *= prior_prob_given_no_shock # [B, K]

        if not mixture.is_batched:
            mean, cov_chol, weights = [p.squeeze(0) for p in (mean, cov_chol, weights)] # [K, F], [K, F, F], [K]

        return mean, cov_chol, weights

    @staticmethod    
    def log_normalizing_factors(
            cov_chol: torch.Tensor,
            weights: torch.Tensor
        ) -> torch.Tensor:
        r"""Compute log-space normalization constants for Gaussian components.

        For each mixture component, computes the log-space normalization factor
        that, combined with the Mahalanobis distance, yields the log-density:

        .. math::

            \log p_k(x) = \text{log_norm}[k] - \frac{1}{2} \|v\|^2,

        where :math:`v = L^{-1}(x - \mu)` and :math:`L` is the Cholesky factor.

        Parameters
        ----------
        cov_chol : torch.Tensor
            Cholesky factors of covariance matrices, shape ``[..., K, F, F]``.
        weights : torch.Tensor
            Mixture weights (including class priors), shape ``[..., K]``.

        Returns
        -------
        torch.Tensor
            Log normalization constants, shape ``[..., K]``. Each entry includes
            the log-determinant of the covariance and the log-weight.
        """
        F = cov_chol.size(-1)
        log_det = 2 * cov_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1) # [..., K]
        log_norm_components = -0.5 * (F * log(2 * torch.pi) + log_det)      # [..., K]
        log_norm = weights.log() + log_norm_components   # [..., K]

        return log_norm


    def broadcast_X(self, X: torch.Tensor, X_contains_batch_dims: bool = True, safety_checks: bool = True) -> torch.Tensor:
        F = X.size(-1)
        if safety_checks:
            if F != self.F:
                raise AssertionError("X has the wrong features dimension")
            if not X.is_floating_point():
                raise AssertionError("X should be a floating point")
            
        len_batch_dims = len(self.batch_dims)
        if X_contains_batch_dims:
            *batch_dims_X, N = X.shape[-(len_batch_dims+2):-1]
            lead_dims = X.shape[:-(len_batch_dims+2)]
            if safety_checks:
                if torch.Size(batch_dims_X)!=self.batch_dims:
                    raise AssertionError("Batch dims does not correspond!!")
            count_to_add_inbetween = 1 # 1 for the mixture component dimension
        else:
            batch_dims_X = []
            *lead_dims, N = X.shape[:-1]
            count_to_add_inbetween = len_batch_dims + 1 # 1 for the mixture component dimension
            
        X = X.to(dtype=self.dtype)
        helper_add_dims = (1,) * count_to_add_inbetween

        X = X.reshape(*lead_dims, *batch_dims_X, *helper_add_dims, N, F) # [*lead_dims, *bb_dims, N, F]
        X = X.movedim(-2, 0) # [N, *lead_dims, *bb_dims, F]
        return X
    
    def mahalanobis_X(self, X_broadcasted) -> torch.Tensor:
        r"""Compute squared Mahalanobis distances to all mixture components.

        For each feature vector and component, solves the triangular system
        :math:`L v = (x - \mu)` where :math:`L` is the Cholesky factor, then
        returns :math:`\|v\|^2`.

        Parameters
        ----------
        X_broadcasted : torch.Tensor
            Broadcasted features of shape ``[*lead_dims, *bb_dims, 1, F]``
            (typically output of :meth:`broadcast_X`). Whereby
            ``bb_dims = [1 for _ in self.batch_dims]``

        Returns
        -------
        torch.Tensor
            Squared Mahalanobis distances of shape ``[*lead_dims, *batch_dims, K]``.

        Notes
        -----
        Although ``torch.linalg.solve_triangular`` with the lower Cholesky factor
        would avoid 
        """
        X_c = X_broadcasted - self.joint_means # [*lead_dims, *batch_dims, K, F]
        X_c = X_c.unsqueeze(-1) # [*lead_dims, *batch_dims, K, F, 1]

        mahal = (
            X_c.mT @             # [*lead_dims, *batch_dims, K, 1, F]
            self.joint_inv_covs @ # [*lead_dims, *batch_dims, K, F, F]
            X_c                  # [*lead_dims, *batch_dims, K, F, 1]
        ) # [*lead_dims, *batch_dims, K, 1, 1]

        return mahal.squeeze(-2, -1) # [*lead_dims, *batch_dims, K]
    
    def log_component_pdf(self, X_broadcasted) -> torch.Tensor:
        r"""Compute log-probability density under each mixture component.

        Parameters
        ----------
        X_broadcasted : torch.Tensor
            Broadcasted features of shape ``[*lead_dims, *batch_dims, 1, F]``.

        Returns
        -------
        torch.Tensor
            Log-densities under each component, shape ``[*lead_dims, *batch_dims, K]``.

        Notes
        -----
        Combines pre-computed normalization factors with the Mahalanobis distance:

        .. math::

            \log \phi_k(x) = \text{log_norm}[k] - \frac{1}{2} M_k(x),

        where :math:`M_k(x)` is the squared Mahalanobis distance.
        """
        mahal_X = self.mahalanobis_X(X_broadcasted) # [*lead_dims, *batch_dims, K]

        log_pdf_per_component = self._log_component_const - (mahal_X/2) # [*lead_dims, *batch_dims, K]

        return log_pdf_per_component # [*lead_dims, *batch_dims, K]
    
    def log_prob_bad_given_no_shock(self, X_broadcasted: torch.Tensor) -> torch.Tensor:
        r"""Compute log posterior of bad outcome (conditioned on no shock).

        Computes the log-space likelihood ratio:

        .. math::

            \log \mathbb{P}(Y=b \mid X) = \log \Phi_b(X) - \log \Phi(X),

        where :math:`\Phi_b(X)` is the marginal likelihood under the bad mixture
        and :math:`\Phi(X)` is the total marginal likelihood.

        Parameters
        ----------
        X_broadcasted : torch.Tensor
            Broadcasted features of shape ``[*lead_dims, *batch_dims, 1, F]``.

        Returns
        -------
        torch.Tensor
            Log posterior probabilities of bad outcome (no shock),
            shape ``[*lead_dims, *batch_dims]``.

        Notes
        -----
        Both marginal likelihoods are computed via ``torch.logsumexp`` to avoid
        numerical underflow.
        """
        log_pdf_per_component = self.log_component_pdf(X_broadcasted) # [*lead_dims, *batch_dims, K]

        # This is equal to \log (\phi_{X_b}(\textt{X})\mathbb{P}(Y=b))
        log_likelihood_X_bad = torch.logsumexp(log_pdf_per_component[..., :self.K_bad], dim=-1) # [*lead_dims, *batch_dims]
        
        # This is equal to \log (\phi_{X_g}(\textt{X})\mathbb{P}(Y=g) +  \phi_{X_b}(\textt{X})\mathbb{P}(Y=b))
        log_marginal_pdf_X = torch.logsumexp(log_pdf_per_component, dim=-1) # [*lead_dims, *batch_dims]

        # \log \mathbb{P}(Y=b | X = \textt{X})
        return log_likelihood_X_bad - log_marginal_pdf_X # [*lead_dims, *batch_dims]
        

    def predict_prob_bad(self, X: torch.Tensor, X_contains_batch_dims: bool = True, safety_checks: bool = True) -> torch.Tensor:
        r"""Compute posterior probability of bad outcome (including shocks if enabled).

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape ``[*lead_dims, F]``.
        safety_checks : bool, default=True
            If ``True``, validates feature dimension and dtype in :meth:`broadcast_X`.

        Returns
        -------
        torch.Tensor
            Posterior probabilities of bad outcome, shape ``[*lead_dims, *batch_dims]``.

        Notes
        -----
        If ``consider_idiosyncratic_shock=True``, incorporates the shock term:

        .. math::

            \mathbb{P}(Y=b \mid X) = (1 - p_s) \mathbb{P}(Y=b \mid X, \text{no shock})
                                    + p_s \mathbb{P}(Y=b \mid \text{shock}),

        Otherwise returns only the shock-free posterior.
        """
        #  *lead_dims, F = X.shape, bb_dims = (1,) * len(self.batch_dims)
        X = self.broadcast_X(X, X_contains_batch_dims, safety_checks) # [*lead_dims, *bb_dims, 1, F]

        # \log \mathbb{P}(Y=b | X = \textt{X})
        log_posterior = self.log_prob_bad_given_no_shock(X) # [*lead_dims, *batch_dims]

        prob_bad = log_posterior.exp()

        if self.consider_idiosyncratic_shock:
            prob_bad *= 1-self.prob_shock
            prob_bad += self.prob_bad_because_of_shock

        if X_contains_batch_dims:
            prob_bad = prob_bad.movedim(0, -1)

        return prob_bad  # [*lead_dims, *batch_dims]
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        r"""Return class probabilities in scikit-learn format.

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix of shape ``[*lead_dims, F]``.

        Returns
        -------
        torch.Tensor
            Class probabilities of shape ``[*lead_dims, *batch_dims, 2]`` where the
            last dimension contains ``[P(Y=0), P(Y=1)]`` (good, bad) respectively.
        """
        prob_bad = self.predict_prob_bad(features)  # [*lead_dims, *batch_dims]
        
        probs = prob_bad.new_empty(prob_bad.shape + (2,))  # [*lead_dims, *batch_dims, gb=2]
        probs[..., 1] = prob_bad
        probs[..., 0] = 1-prob_bad

        return probs # [*lead_dims, *batch_dims, gb=2]

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        r"""Predict the most likely class.

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix of shape ``[*lead_dims, F]``.

        Returns
        -------
        torch.Tensor
            Binary class predictions (0 or 1) of shape ``[*lead_dims, *batch_dims]``.
        """
        probs = self.predict_proba(features)  # [..., *batch_dims, 2]
        return torch.argmax(probs, dim=-1)  # [..., *batch_dims]

    def decision_function(self, features: torch.Tensor) -> torch.Tensor:
        r"""Compute the log-odds decision scores.

        Returns the log-odds (log posterior odds) of the bad outcome:

        .. math::

            z(x) = \log \frac{\mathbb{P}(Y=b \mid x)}{\mathbb{P}(Y=g \mid x)}.

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix of shape ``[*lead_dims, F]``.

        Returns
        -------
        torch.Tensor
            Log-odds of shape ``[*lead_dims, *batch_dims]``. Positive values favor
            the bad outcome, negative values favor the good outcome.

        Notes
        -----
        This is the unnormalized decision score that represents the model's
        confidence in the bad outcome. It is monotonic in the posterior probability
        and is typically used for ROC curves, calibration analysis, and threshold tuning.
        """
        X = self.broadcast_X(features, safety_checks=True)
        return self.log_prob_bad_given_no_shock(X)

    def to_state_dict(self) -> Dict[str, ndarray]:
        r"""Return a lightweight, device-agnostic snapshot of the model state.

        Returns
        -------
        dict
            Dictionary with the following keys:

            ``"joint_means"``
                ``numpy.ndarray`` of shape ``[*batch_dims, K, F]``.
            ``"joint_cov_chols"``
                ``numpy.ndarray`` of shape ``[*batch_dims, K, F, F]``.
            ``"joint_weights"``
                ``numpy.ndarray`` of shape ``[*batch_dims, K]``.
            ``"prob_bad_given_no_shock"``
                ``float``.
            ``"prob_shock"``
                ``float`` or ``None``.
            ``"prob_bad_because_of_shock"``
                ``float`` or ``None``.
            ``"K_bad"``
                ``int``.
            ``"is_batched"``
                ``bool``.

        Notes
        -----
        The returned arrays are detached CPU copies; modifying them does not
        affect the model.  The dictionary is fully pickle-safe and device-independent.
        """
        return {
            "joint_means": self.joint_means.detach().cpu().numpy(),
            "joint_cov_chols": self.joint_cov_chols.detach().cpu().numpy(),
            "joint_weights": self.joint_weights.detach().cpu().numpy(),
            "prob_bad_given_no_shock": float(self.prob_bad_given_no_shock),
            "prob_shock": float(self.prob_shock) if self.prob_shock is not None else None,
            "prob_bad_because_of_shock": float(self.prob_bad_because_of_shock) if self.prob_bad_because_of_shock is not None else None,
            "_K_bad": int(self._K_bad),
            "is_batched": bool(self._is_batched),
        }

    def load_from_state_dict(self, state_dict: Dict[str, Union[ndarray, torch.Tensor]]) -> "PerfectBayesClassifier":
        r"""Restore model state from a snapshot produced by :meth:`to_state_dict`.

        Parameters
        ----------
        state_dict : dict
            Must contain exactly the keys produced by :meth:`to_state_dict`.

        Returns
        -------
        PerfectBayesClassifier
            Returns ``self`` for method chaining.

        Raises
        ------
        ValueError
            If required keys are missing, unexpected keys are present, or shapes
            do not match.
        """
        expected = {
            "joint_means", "joint_cov_chols", "joint_weights",
            "prob_bad_given_no_shock", "prob_shock", "prob_bad_because_of_shock",
            "_K_bad", "is_batched",
        }
        given = set(state_dict.keys())
        if given != expected:
            raise ValueError(
                f"state_dict keys mismatch.\n  Expected: {sorted(expected)}\n  Got: {sorted(given)}"
            )

        # Validate and convert numpy arrays to tensors
        for key in ["joint_means", "joint_cov_chols", "joint_weights"]:
            data = state_dict[key]
            if isinstance(data, ndarray):
                data = torch.from_numpy(data)
            elif not isinstance(data, torch.Tensor):
                raise ValueError(
                    f"'{key}' must be a numpy.ndarray or torch.Tensor, got {type(data)}"
                )
            if data.shape != getattr(self, key).shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': expected {tuple(getattr(self, key).shape)}, "
                    f"got {tuple(data.shape)}"
                )

        # Restore tensors
        with torch.no_grad():
            self.joint_means.copy_(
                state_dict["joint_means"].to(
                    dtype=self.joint_means.dtype, device=self.joint_means.device
                )
            )
            self.joint_cov_chols.copy_(
                state_dict["joint_cov_chols"].to(
                    dtype=self.joint_cov_chols.dtype, device=self.joint_cov_chols.device
                )
            )
            self.joint_weights.copy_(
                state_dict["joint_weights"].to(
                    dtype=self.joint_weights.dtype, device=self.joint_weights.device
                )
            )
            # Recompute log normalization factors, considering maybe new device and dtype
            self._log_component_const.copy_(
                self.log_normalizing_factors(self.joint_cov_chols, self.joint_weights)
            )

        # Restore scalar attributes
        self.prob_bad_given_no_shock = float(state_dict["prob_bad_given_no_shock"])
        self.prob_shock = float(state_dict["prob_shock"]) if state_dict["prob_shock"] is not None else None
        self.prob_bad_because_of_shock = float(state_dict["prob_bad_because_of_shock"]) if state_dict["prob_bad_because_of_shock"] is not None else None

        return self

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "PerfectBayesClassifier":
        r"""Move all tensors to a target device and/or data type.

        Parameters
        ----------
        device : torch.device, optional
            Target device. If ``None``, device is not changed.
        dtype : torch.dtype, optional
            Target data type. If ``None``, dtype is not changed.

        Returns
        -------
        PerfectBayesClassifier
            Returns ``self`` for method chaining.

        Notes
        -----
        All mixture tensors and pre-computed log normalization factors are moved
        together to ensure consistency.
        """
        if device is not None or dtype is not None:
            with torch.no_grad():
                if device is not None and dtype is not None:
                    self.joint_means.data = self.joint_means.to(device=device, dtype=dtype)
                    self.joint_cov_chols.data = self.joint_cov_chols.to(device=device, dtype=dtype)
                    self.joint_weights.data = self.joint_weights.to(device=device, dtype=dtype)
                elif device is not None:
                    self.joint_means.data = self.joint_means.to(device=device)
                    self.joint_cov_chols.data = self.joint_cov_chols.to(device=device)
                    self.joint_weights.data = self.joint_weights.to(device=device)
                else:  # dtype is not None
                    self.joint_means.data = self.joint_means.to(dtype=dtype)
                    self.joint_cov_chols.data = self.joint_cov_chols.to(dtype=dtype)
                    self.joint_weights.data = self.joint_weights.to(dtype=dtype)
                
                # Recompute log normalization factors on the new device/dtype
                self._log_component_const.data = self.log_normalizing_factors(
                    self.joint_cov_chols, self.joint_weights
                )

        return self
    