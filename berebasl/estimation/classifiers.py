import inspect

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Optional

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
            train_model : Callable[["model", torch.Tensor, torch.Tensor], None]
    ):
        self.model = model
        self.predict_model_probs = predict_model_probs
        self.train_model = train_model

    def fit(self, features : torch.Tensor, labels : torch.Tensor) -> None:
        self.train_model(self.model, features, labels)

    def predict_proba(self, features : torch.Tensor) -> torch.Tensor:
        # with features.shape = [..., k], it should
        # return a tensor of shape [..., 2] containing the predicted
        # probabilities of label = 0, 1 respectively along the last axis
        return self.predict_model_probs(self.model, features)

    @staticmethod
    def obj_has_needed_funs(obj):
        expected_funs_with_param_count = [
            ("fit", 2),
            ("predict_proba", 1)
        ]
        for fun_name, param_count in expected_funs_with_param_count:
            if not hasattr(obj, fun_name):
                return False
            
            if not function_has_expected_signature(getattr(obj, fun_name), param_count):
                return False
            
        return True

class TorchLogistic(nn.Module):
    r"""
    Multinomial or binary logistic regression implemented in PyTorch using full-batch
    L-BFGS optimization. This class is designed to mimic the behavior of classical
    GLM-style maximum likelihood estimation (MLE), including the binary Logit model
    and the multinomial softmax model.

    The model supports inputs of arbitrary leading shape ``[...]`` as long as the
    final dimension corresponds to ``n_features``. All leading dimensions are treated
    as part of a single flattened batch during optimization.

    .. warning::
        The ``fit`` method uses PyTorch's :class:`torch.optim.LBFGS`, which performs
        **multiple internal optimization steps** inside a single call to
        ``optimizer.step``. This is *not* equivalent to a single gradient update as in
        SGD/Adam. Instead, L-BFGS repeatedly evaluates the closure, performs line
        searches, and updates curvature estimates until convergence criteria are met.
        Users should not assume epoch-like behavior.

    .. warning::
        Because L-BFGS requires full-batch gradients, all leading dimensions of ``X``
        and ``y`` are flattened into one effective batch. For example, inputs of shape
        ``[B, T, n_features]`` are treated as a batch of size ``B*T``. This is correct
        for MLE but may be surprising if the user expects per-group optimization.

    Parameters
    ----------
    n_features : int
        Number of input features (size of the last dimension of ``X``).

    n_classes : int, default=2
        Number of classes. If ``n_classes == 2``, the model uses a single-logit
        Bernoulli formulation with a sigmoid link. If ``n_classes > 2``, the model
        uses a ``n_classes``-logit softmax formulation.

    lbfgs_kwargs : dict, optional
        Keyword arguments forwarded directly to :class:`torch.optim.LBFGS`. If
        ``None``, a set of defaults approximating scikit-learn's behavior is used.
        Keys are validated against the actual L-BFGS constructor when
        ``secure_init=True``.

    secure_init : bool, default=True
        If ``True``, performs argument validation and safety checks. Set to ``False``
        only if you know exactly what you are doing.
    """

    def __init__(
            self, 
            n_features : int, 
            n_classes : int = 2,
            lbfgs_kwargs : Optional[dict] = None,
            secure_init : bool = True
        ):
        r"""
        Initialize the logistic regression model.

        Notes
        -----
        For ``n_classes == 2``, the model outputs a single logit ``z`` corresponding
        to the log-odds:

        .. math::

            z = \log \frac{p(y=1 \mid x)}{p(y=0 \mid x)}

        For ``n_classes > 2``, the model outputs ``n_classes`` logits and applies a
        softmax during probability prediction.
        """
        super().__init__()
        self.n_classes = int(n_classes)
        if lbfgs_kwargs is None:
            lbfgs_kwargs = {
                'lr' : 1,
                'max_iter' :  100,
                'tolerance_grad' :  torch.finfo(torch.get_default_dtype()).eps * 64,
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
            dtype=torch.get_default_dtype()
        )

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
        if self.n_classes == 2:
            return logits.flatten(-2,-1)
        return logits
    
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
        if self.n_classes == 2:
            p1 = torch.sigmoid(logits).unsqueeze(-1) # [..., 1]
            p0 = 1 - p1 # [..., 1]
            probs = torch.cat([p0, p1], dim=-1) # [..., 2]
        else:
            probs = torch.softmax(logits, dim=-1) # [..., n_classes]

        return probs # [..., n_classes]
    
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
        - The optimization is deterministic only if the user controls PyTorch's
          random seeds and uses deterministic linear algebra kernels.
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