from copy import deepcopy
import inspect
from math import sqrt

from numpy import ndarray
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Dict, Optional, Union

from berebasl.utils.tensor_validation import assert_tensors

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



class BatchedLogistic(nn.Module):
    r"""
    Batched binary logistic regression fitted via exact Newton-Raphson iterations,
    equivalent to Iteratively Re-weighted Least Squares (IRLS) for the logit link
    along each batch of Observations.

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
    for all subsequent iterations (early stopping per element).  An external boolean
    mask can additionally exclude selected elements from training from iteration zero.

    

    Attributes
    ----------
    beta : torch.Tensor
        Coefficient buffer of shape ``[*batch_shape, p_aug, 1]``, where
        ``p_aug = n_features + 1`` when ``fit_intercept=True`` and ``n_features``
        otherwise.  When ``fit_intercept=True`` the first row of every coefficient
        vector corresponds to the intercept (consistent with the statistics
        convention of placing the constant in the first column of the design
        matrix).
    batch_shape : torch.Size
    n_features : int
    fit_intercept : bool
    min_iter : int
    max_iter : int
    tol_score : float
    tol_step : float

    Methods
    -------
    fit(X, y, mask=None)
        Fit all batch elements simultaneously via batched Newton-Raphson.
    forward(X)
        Compute raw logits for all batch elements.
    predict_proba(X)
        Return ``[p(y=0), p(y=1)]`` probabilities for all batch elements.
    predict(X)
        Return binary class predictions for all batch elements.
    decision_function(X)
        Alias for :meth:`forward`.
    reset_beta()
        Re-initialise ``beta`` to zero in-place.
    to_state_dict()
        Lightweight, device-agnostic parameter snapshot.
    load_from_state_dict(state_dict)
        Restore parameters from a snapshot produced by :meth:`to_state_dict`.
    instantiate_from_state_dict(state_dict, device=None)
        Class method: construct a new instance from a snapshot.
    from_torch_logistic(torch_logistic, batch_shape, ...)
        Static method: construct a :class:`BatchedLogistic` from a fitted
        :class:`TorchLogistic` instance, inheriting its structural configuration.

    Notes
    -----
    **Why ``bmm`` throughout?**
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
    ):
        """
        Initialize the class

        Parameters
        ----------
        n_features : int
            Number of input features (excluding the implicit intercept if any).
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
            Prevents premature termination near the (trivial) zero initialisation.
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
        """
        super().__init__()

        self.n_features    = int(n_features)
        self.batch_shape   = torch.Size(batch_shape)
        self.fit_intercept = bool(fit_intercept)
        self.min_iter      = int(min_iter)
        self.max_iter      = int(max_iter)

        _dtype = dtype if dtype is not None else torch.get_default_dtype()
        _eps   = torch.finfo(_dtype).eps

        self.tol_score = float(tol_score) if tol_score is not None else float(_eps ** (2 / 3))
        self.tol_step  = float(tol_step)  if tol_step  is not None else float(_eps ** (7 / 8))
        self._tikhonov = float(_eps)

        # p_aug: number of rows in beta per batch element.
        # With fit_intercept the design matrix is [1 | X] (intercept first,
        # following the statistics convention), so beta[..., 0, :] is the intercept.
        self._p_aug = self.n_features + 1 if self.fit_intercept else self.n_features

        beta_shape = self.batch_shape + (self._p_aug, 1)
        self.register_buffer(
            "beta",
            torch.zeros(beta_shape, dtype=_dtype, device=device),
        )

        # _augment is resolved once at init, mirroring the logit_to_probs
        # pattern in TorchLogistic, to avoid branching at call time.
        if self.fit_intercept:
            self._augment: Callable[[torch.Tensor], torch.Tensor] = self._prepend_ones
        else:
            self._augment = lambda X: X

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepend_ones(X: torch.Tensor) -> torch.Tensor:
        r"""
        Prepend a column of ones to ``X`` to absorb the intercept into ``beta``.

        Placing the constant in the **first** column follows the statistics
        convention (e.g. as in R's model matrices or classical GLM textbooks),
        so ``beta[..., 0, :]`` always corresponds to the intercept term.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]``.

        Returns
        -------
        torch.Tensor
            Shape ``[*batch_shape, N, n_features + 1]`` with a leading ones column.
        """
        ones = torch.ones(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)
        return torch.cat([ones, X], dim=-1)    # [..., N, p_aug], ones first

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        r"""
        Device on which ``beta`` - the only internal tensor - resides.

        Returns
        -------
        torch.device
        """
        return self.beta.device

    # ------------------------------------------------------------------
    # Forward / prediction
    # ------------------------------------------------------------------

    def validate_inputs(self, feats: torch.Tensor, lbls: Optional[torch.Tensor]):
        if lbls is not None:
            assert_tensors(
                feats, lbls, tensor_names="feats, lbls",
                checks=["are_tensors", "same_device"]
            )
            if feats.shape[:-1] != lbls.shape:
                raise AssertionError(
                    "feats and lbls do not have the right dimensions"
            )
        assert_tensors(
            feats, self.betas, tensor_names="feats, betas",
            checks=["same_dtype", "same_device"]
        )
        if feats.dim() < self.betas.dim():
            raise AssertionError(
                "feats does not have enough dimensions"
            )
        
        if feats.size(-1) != self.betas.size(-2):
            raise AssertionError("feats has the wrong amount of features")
        
        middle_dims_feats_as_tensor = torch.tensor(feats.shape[-self.betas.dim():-2])
        
        middle_feats_dims_ok = (
            (middle_dims_feats_as_tensor == torch.tensor(self.betas.shape[:-2])) |
            middle_dims_feats_as_tensor == 1
        )
        
        if not torch.all(middle_feats_dims_ok):
            raise AssertionError("Wrong dimensions")
        
    def handle_shapes(self, feats: torch.Tensor, lbls: Optional[torch.Tensor]):
        extra_batch_dims_feats = feats.shape[:-betas.dim()]
        feats = feats.expand(*extra_batch_dims_feats, *betas.shape[:-2], *feats.shape[-2:])
        lbls = lbls.expand(*extra_batch_dims_feats, *betas.shape[:-2], feats.size(-2))
        betas = betas.expand(*extra_batch_dims_feats, *betas.shape)

        feats = feats.view(-1, feats.shape[-2:])
        lbls = feats.view(-1, feats.size(-2), 1)
        betas = betas.view(-1, betas.shape[-2:])

        return feats, lbls, betas, extra_batch_dims_feats

    def forward(
            self, 
            X: torch.Tensor, 
            betas_reshaped: Optional[torch.Tensor], 
            extra_batch_dims_feats: Optional[torch.Size]
        ) -> torch.Tensor:
        r"""
        Compute raw logits for all batch elements.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``[*batch_shape, N, n_features]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[*batch_shape, N]``.

        Notes
        -----
        The logit for observation :math:`i` in batch element :math:`b` is

        .. math::

            z_{b,i} = \tilde{x}_{b,i}^{\top} \beta_b,

        where :math:`\tilde{x}_{b,i}` includes the prepended one when
        ``fit_intercept=True``.  The computation uses ``torch.bmm`` over the
        flattened batch dimension so that independence across batch elements is
        explicit and verifiable.
        """
        if (betas_reshaped is None) ^ (extra_batch_dims_feats is None):
            raise RuntimeError("Either pass none of betas_reshaped and extra_batch_dims or pass both")
        X_aug = self._augment(X)               # [..., N, p_aug]

        # Flatten all leading batch dims into one axis for bmm.
        B     = self.beta.shape[:-2].numel()
        N     = X_aug.shape[-2]

        X_flat    = X_aug.reshape(B, N, self._p_aug)       # [B, N, p_aug]
        beta_flat = self.beta.reshape(B, self._p_aug, 1)   # [B, p_aug, 1]

        logits_flat = torch.bmm(X_flat, beta_flat)         # [B, N, 1]
        return logits_flat.reshape(self.batch_shape + (N,))  # [..., N]

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
            ``[p(y=0), p(y=1)]``.
        """
        p1 = torch.sigmoid(self.forward(X)).unsqueeze(-1)  # [..., N, 1]
        return torch.cat([1.0 - p1, p1], dim=-1)           # [..., N, 2]

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
        """
        with torch.no_grad():
            self.beta.zero_()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        mask : torch.Tensor, optional
            Boolean tensor of shape ``[*batch_shape]``.  Elements where ``mask``
            is ``True`` are **active** and will be optimised.  Elements where
            ``mask`` is ``False`` are frozen from step 0 — their ``beta`` is
            never modified.  If ``None``, all elements are treated as active.

        Returns
        -------
        self : BatchedLogistic
            The fitted model (``self.beta`` updated in-place via ``add_``).

        Notes
        -----
        **Per-element independence**

        All matrix products use ``torch.bmm`` over the explicitly flattened
        batch dimension ``B = prod(batch_shape)`` so that each batch element's
        computation is provably independent.  The Hessian ``H`` and score ``s``
        for element ``b`` depend only on ``X[b]``, ``y[b]``, and ``beta[b]``.

        **Newton step**

        .. math::

            H_t &= \tilde{X}^{\top} W_t \tilde{X} + \lambda I, \\
            s_t &= \tilde{X}^{\top}(y - \hat{p}_t), \\
            \Delta\beta_t &= H_t^{-1} s_t
                \quad\text{(Cholesky solve)},

        where :math:`\lambda = \varepsilon_{\text{machine}}` is the Tikhonov
        guard that ensures :math:`H_t` is numerically SPD even under perfect
        separation or near-multicollinearity, without meaningfully biasing the
        MLE.

        **Convergence and early stopping**

        A batch element is declared converged when

        .. math::

            \|s_t\|_\infty < \texttt{tol\_score}
            \;\text{ and }\;
            \|\Delta\beta_t\|_\infty < \texttt{tol\_step}.

        The combined criterion is necessary because near-multicollinearity can
        produce a small score norm while the Cholesky solve still yields large
        parameter steps.  Once frozen, an element's update is zeroed out via
        the active mask before the in-place ``add_``, so ``beta`` is never
        modified after convergence.  The ``min_iter`` guard prevents premature
        freezing near the zero initialisation.
        """
        self.reset_beta()

        X_aug = self._augment(X)                           # [..., N, p_aug]
        y_col = y.unsqueeze(-1).to(self.beta.dtype)        # [..., N, 1]

        # Flatten all leading batch dims into a single axis B for bmm.
        B     = self.beta.shape[:-2].numel()
        N     = X_aug.shape[-2]
        p_aug = self._p_aug

        X_flat = X_aug.reshape(B, N, p_aug)                # [B, N, p_aug]
        y_flat = y_col.reshape(B, N, 1)                    # [B, N, 1]

        # Tikhonov term: built once, never changes.
        # Shape [B, p_aug, p_aug] — eps * I per batch element.
        tikhonov = self._tikhonov * torch.eye(
            p_aug, dtype=self.beta.dtype, device=self.beta.device
        ).unsqueeze(0).expand(B, -1, -1)

        # active_flat[b] = True  →  element b should still be updated
        if mask is not None:
            active_flat = mask.reshape(B).to(device=self.beta.device)  # [B], bool
        else:
            active_flat = torch.ones(B, dtype=torch.bool, device=self.beta.device)

        with torch.no_grad():
            for iteration in range(self.max_iter):

                beta_flat = self.beta.reshape(B, p_aug, 1)  # [B, p_aug, 1]

                # ---- forward pass ----------------------------------------
                logits = torch.bmm(X_flat, beta_flat)        # [B, N, 1]
                p_hat  = torch.sigmoid(logits)               # [B, N, 1]

                # ---- IRLS weights ----------------------------------------
                W = p_hat * (1.0 - p_hat)                    # [B, N, 1]

                # ---- score  s = X^T (y - p_hat) --------------------------
                residual = y_flat - p_hat                     # [B, N, 1]
                score = torch.bmm(
                    X_flat.transpose(-2, -1), residual
                )                                            # [B, p_aug, 1]

                # ---- Hessian  H = X^T W X + lambda I ---------------------
                # Weight rows of X by W then use bmm: avoids the dense
                # [B, N, N] diagonal weight matrix entirely.
                X_w = X_flat * W                             # [B, N, p_aug]
                H   = torch.bmm(
                    X_flat.transpose(-2, -1), X_w
                ) + tikhonov                                 # [B, p_aug, p_aug]

                # ---- Newton step via Cholesky solve ----------------------
                # H is SPD by construction (Tikhonov guard ensures this even
                # under separation or multicollinearity).
                # cholesky_solve avoids squaring kappa(H) vs. explicit inversion.
                L          = torch.linalg.cholesky(H)       # [B, p_aug, p_aug]
                delta_beta = torch.cholesky_solve(score, L) # [B, p_aug, 1]

                # ---- convergence check  (L-inf on score AND step) --------
                score_linf = score.abs().amax(dim=(-2, -1))       # [B]
                step_linf  = delta_beta.abs().amax(dim=(-2, -1))  # [B]

                newly_converged = (
                    (score_linf < self.tol_score) &
                    (step_linf  < self.tol_step)
                )

                # Honour min_iter: do not freeze before that threshold
                if iteration >= self.min_iter:
                    active_flat = active_flat & ~newly_converged

                if not active_flat.any():
                    break

                # ---- masked in-place parameter update --------------------
                # active_w[b, 0, 0] = 1.0 if active else 0.0
                # Broadcasts over [B, p_aug, 1] — frozen elements get zero delta.
                active_w = active_flat.to(self.beta.dtype).reshape(B, 1, 1)
                masked_delta = (delta_beta * active_w).reshape(
                    self.batch_shape + (p_aug, 1)
                )
                # In-place add preserves the buffer registration.
                self.beta.add_(masked_delta)

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
            Dictionary with keys:

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
            not match, or ``"beta"`` is neither a NumPy array nor a PyTorch
            tensor.
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

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_torch_logistic(
        torch_logistic,
        batch_shape: Union[torch.Size, Tuple[int, ...]],
        min_iter: int = 3,
        max_iter: int = 100,
        device: Optional[torch.device] = None,
    ) -> "BatchedLogistic":
        r"""
        Construct a :class:`BatchedLogistic` from a :class:`TorchLogistic` instance.

        Structural configuration (``n_features``, ``fit_intercept``, dtype,
        tolerances) is read directly from ``torch_logistic``.  Tolerances are
        inherited from ``lbfgs_kwargs`` when present (``tolerance_grad`` →
        ``tol_score``, ``tolerance_change`` → ``tol_step``); otherwise the
        dtype-derived defaults are used.  ``beta`` is initialised to zero because
        the batch shape introduces new independent coefficient vectors that have no
        correspondence to the single vector in :class:`TorchLogistic`.

        Parameters
        ----------
        torch_logistic : TorchLogistic
            A binary (``n_classes == 2``) :class:`TorchLogistic` instance.
        batch_shape : torch.Size or tuple of int
            Leading batch dimensions for the new instance.
        min_iter : int, default=3
            Minimum Newton iterations.
        max_iter : int, default=100
            Maximum Newton iterations.
        device : torch.device, optional
            Target device.  Defaults to the device of ``torch_logistic``.

        Returns
        -------
        BatchedLogistic

        Raises
        ------
        AssertionError
            If ``torch_logistic.n_classes != 2``.
        """
        assert torch_logistic.n_classes == 2, (
            "from_torch_logistic only supports binary TorchLogistic instances "
            f"(n_classes=2), got n_classes={torch_logistic.n_classes}."
        )

        n_features    = torch_logistic.lin_estimator.in_features
        fit_intercept = torch_logistic.lin_estimator.bias is not None
        dtype         = torch_logistic.lin_estimator.weight.dtype
        _eps          = torch.finfo(dtype).eps

        lbfgs_kw  = getattr(torch_logistic, "lbfgs_kwargs", {})
        tol_score = float(lbfgs_kw.get("tolerance_grad",   _eps ** (2 / 3)))
        tol_step  = float(lbfgs_kw.get("tolerance_change", _eps ** (7 / 8)))

        if device is None:
            device = torch_logistic.lin_estimator.weight.device

        return BatchedLogistic(
            n_features    = n_features,
            batch_shape   = batch_shape,
            fit_intercept = fit_intercept,
            min_iter      = min_iter,
            max_iter      = max_iter,
            tol_score     = tol_score,
            tol_step      = tol_step,
            device        = device,
            dtype         = dtype,
        )