from copy import deepcopy
import inspect
from math import sqrt

from numpy import ndarray
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Dict, Optional, Union

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
