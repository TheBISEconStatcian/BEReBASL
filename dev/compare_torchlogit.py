# ============================
# Standard library imports
# ============================
import time
import warnings

from typing import Tuple, Optional

# ============================
# Third-party imports
# ============================
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLMResults
import statsmodels.api as sm
from statsmodels.genmod import families
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import torch

# ============================
# Local project imports
# ============================
from berebasl.estimation.classifiers import BatchedLogistic, TorchLogistic

# ============================
# Global configuration
# ============================
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6)
# Supurious warning which was corrected in github repo but not
# in the current sci-kit learn release
warnings.filterwarnings(
    "ignore",
    message="Setting penalty=None will ignore the C and l1_ratio parameters"
)

def fit_glm(
    X: np.ndarray,
    label: np.ndarray,
    X_vali: Optional[np.ndarray] = None
) -> Tuple[GLMResults, np.ndarray, float, float]:
    r"""
    Fit a classical GLM logistic regression model using ``statsmodels`` and measure
    both training and inference time.

    This function performs a binomial GLM fit with a logit link, adds an intercept
    column automatically, and returns predicted probabilities in the standard
    two-column format ``[p(y=0), p(y=1)]``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, d)``. No intercept column is required; one is
        added internally via :func:`statsmodels.api.add_constant`.

    label : np.ndarray
        Binary target vector of shape ``(N,)`` containing values ``0`` or ``1``.
        Values are passed directly to ``statsmodels`` without modification.

    Returns
    -------
    (fitted_glm, X_glm) : tuple
        ``fitted_glm`` is the :class:`statsmodels.genmod.generalized_linear_model.GLMResults`
        object. ``X_glm`` is the augmented design matrix with an added intercept.

    glm_probs : np.ndarray
        Predicted probabilities of shape ``(N, 2)``, where the columns correspond to
        ``[p(y=0), p(y=1)]``.

    train_time : float
        Wall-clock time (in seconds) required to fit the GLM.

    inference_time : float
        Wall-clock time (in seconds) required to compute predicted probabilities.

    Notes
    -----
    - ``statsmodels`` performs full maximum-likelihood estimation using iteratively
      reweighted least squares (IRLS).
    - The GLM fit is deterministic given fixed inputs.
    """
    begin_glm = time.time()
    X_glm = sm.add_constant(X)
    glm_lr = sm.GLM(
        endog=label,
        exog=X_glm,
        family=families.Binomial()
    )
    fitted_glm = glm_lr.fit()
    end_glm = time.time()

    train_time = end_glm - begin_glm

    begin_inference = time.time()
    glm_probs = fitted_glm.predict(X_glm if X_vali is None else sm.add_constant(X_vali))
    glm_probs = np.column_stack([1 - glm_probs, glm_probs])
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return fitted_glm, glm_probs, train_time, inference_time


def fit_sklearn(
    X: np.ndarray,
    label: np.ndarray
) -> Tuple[LogisticRegression, np.ndarray, float, float]:
    r"""
    Fit a scikit-learn logistic regression model using ``lbfgs`` and measure
    training and inference time.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, d)``.

    label : np.ndarray
        Target vector of shape ``(N,)``. May contain binary or multiclass labels.

    Returns
    -------
    sk_lr : LogisticRegression
        The fitted scikit-learn logistic regression estimator.

    sk_probs : np.ndarray
        Predicted probabilities of shape ``(N, C)``, where ``C`` is the number of
        classes inferred from ``label``.

    train_time : float
        Wall-clock time (in seconds) required to fit the model.

    inference_time : float
        Wall-clock time (in seconds) required to compute predicted probabilities.

    Notes
    -----
    - ``C=np.inf`` disables regularization, making the estimator closer to a pure
      maximum-likelihood logistic regression.
    - ``lbfgs`` is a quasi-Newton optimizer similar to the one used in the Torch
      implementation.
    """
    begin_sklearn = time.time()
    sk_lr = LogisticRegression(C=np.inf, l1_ratio=0, solver="lbfgs")
    sk_lr = sk_lr.fit(X, label)
    end_sklearn = time.time()

    train_time = end_sklearn - begin_sklearn

    begin_inference = time.time()
    sk_probs = sk_lr.predict_proba(X)
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return sk_lr, sk_probs, train_time, inference_time


def fit_torch(
    X_torch: torch.Tensor,
    label_torch: torch.Tensor,
    X_vali: Optional[torch.Tensor]=None
) -> Tuple[TorchLogistic, torch.Tensor, float, float]:
    r"""
    Fit the custom :class:`TorchLogistic` model using full-batch L-BFGS and measure
    training and inference time.

    Parameters
    ----------
    X_torch : torch.Tensor
        Feature tensor of shape ``(N, d)`` or with arbitrary leading dimensions
        ``[..., d]``. All leading dimensions are flattened during optimization.

    label_torch : torch.Tensor
        Target tensor. For binary classification, may have shape ``[..., 1]`` or
        ``[...]``. For multiclass classification, must contain integer labels in
        ``{0, ..., C-1}``.

    Returns
    -------
    torch_lr : TorchLogistic
        The fitted TorchLogistic model.

    torch_probs : torch.Tensor
        Predicted probabilities of shape ``[..., C]`` where ``C`` is the number of
        classes inferred from ``label_torch``.

    train_time : float
        Wall-clock time (in seconds) required to fit the model.

    inference_time : float
        Wall-clock time (in seconds) required to compute predicted probabilities.

    Notes
    -----
    - The underlying optimizer is PyTorch's full-batch L-BFGS, which performs
      multiple internal evaluations of the closure per call to ``step``.
    - All leading dimensions of ``X_torch`` and ``label_torch`` are treated as a
      single flattened batch during optimization.
    - The model is evaluated in ``eval`` mode during inference.
    """
    begin_torch = time.time()
    torch_lr = TorchLogistic(
        n_features=X_torch.shape[-1],
        n_classes=int(label_torch.max().item() + 1),
        device=X_torch.device
    )
    torch_lr.fit(X_torch, label_torch, reduction='sum')
    end_torch = time.time()

    train_time = end_torch - begin_torch

    begin_inference = time.time()
    torch_lr.eval()
    torch_probs = torch_lr.predict_proba(X_torch if X_vali is None else X_vali)
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return torch_lr, torch_probs.detach(), train_time, inference_time

def fit_batched_logistic(
        Xs_torch: torch.Tensor,
        labels_torch: torch.Tensor,
        mask_valid_obs: torch.Tensor,
        X_vali: Optional[torch.Tensor]=None
) -> Tuple[BatchedLogistic, torch.Tensor, float, float]:
    begin_torch = time.time()
    batched_lr = BatchedLogistic(
        n_features=Xs_torch.size(-1),
        batch_shape=labels_torch.shape[:-1],
        device=Xs_torch.device
    )
    batched_lr.fit(Xs_torch, labels_torch, mask_valid_obs)
    end_torch = time.time()

    train_time = end_torch - begin_torch

    begin_inference = time.time()
    batched_lr.eval()
    batched_probs = batched_lr.predict_proba(Xs_torch if X_vali is None else X_vali)
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return batched_lr, batched_probs.detach(), train_time, inference_time


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data          # shape (150, 4)
    X_torch = torch.from_numpy(X)
    y = iris.target.astype(float)        # shape (150,)
    feature_names = iris.feature_names
    target_names = iris.target_names

    wished_target = "versicolor"


    y_bin = (y == np.where(target_names==wished_target)[0].item()).astype(float)

    for label, desc in [(y_bin, "binary"), (y, "multiclass")]:
        is_binary = desc == "binary"
        label_torch = torch.from_numpy(label)
        label_torch = label_torch.unsqueeze(-1) if is_binary else label_torch.to(int)
        print("\nBeginning", desc, "comparision:")

        print("\tStep 0: Estimation")
        if is_binary: # This gives an error
            print("\t\t0. GLM:")
            fitted_glm, glm_probs, train_time, _ = fit_glm(X, label)

            print("\t\t\tTime needed:", round(train_time*1e3, 2), "(ms)")

            glm_probs = torch.from_numpy(glm_probs)

        print("\t\t1. scikit learn")
        sk_lr, sk_probs, train_time, _ = fit_sklearn(X, label)
        sk_probs = torch.from_numpy(sk_probs)
        print("\t\t\tTime needed:", round(train_time*1e3, 2), "(ms)")

        print("\t\t2. torch implementation")
        torch_lr, torch_probs, train_time, _ = fit_torch(X_torch, label_torch)
        print("\t\t\tTime needed:", round(train_time*1e3, 2), "(ms)")

        print("\tStep 1: Parameter Comparision. Order (GLM), Sklearn, torch")
        print("\t\tComparision of weights")
        if is_binary:
            joint_weights = torch.cat([
                torch.from_numpy(fitted_glm.params[1:]).unsqueeze(0), 
                torch.from_numpy(sk_lr.coef_), 
                torch_lr.lin_estimator.weight.detach()
            ])
        else:
            joint_weights = torch.stack([
                torch.from_numpy(sk_lr.coef_), 
                torch_lr.lin_estimator.weight.detach()
            ], dim = 1)
        print(joint_weights)

        print("\n\t\tComparision of intercept")
        print(torch.stack(
            ([torch.from_numpy(fitted_glm.params[[0]])] if is_binary else []) + [
                torch.from_numpy(sk_lr.intercept_), torch_lr.lin_estimator.bias.detach()
            ]
            ))


        print("\tStep 2: Comparision of predicted probabilities.")
        joint_probs = torch.stack(
                ([glm_probs] if is_binary else [])+ [sk_probs, torch_probs], 
                dim=1
            )

        print("\t\tRandomly chosen probs:")
        print(joint_probs[torch.randint(0, sk_probs.shape[0], size=(6,))])
        print("\t\tAbsolute Distances: (mean and quantiles=.01,.25,.5,.75,.99)")
        mae = lambda a, b : (a-b).abs().mean().detach().item()
        ae_quantiles = lambda a, b, q=torch.tensor([0.01,0.25,0.5,.75,.99]): (a - b).abs().quantile(q)

        comp_spec = [
            ((sk_probs, torch_probs), "scikit vs. torch:")
        ]
        if is_binary:
            comp_spec += [
                ((sk_probs, glm_probs), "scikit vs. GLM:"),
                ((glm_probs, torch_probs), "GLM vs. torch:"),
                ((glm_probs, sk_probs), "GLM vs. scikit:"),
            ]

        for pair, desc_pair in comp_spec:
            print('\t'*3, desc_pair)
            for f in [mae, ae_quantiles]:
                print('\t'*4, f(*pair))
        
        print("\tStep 3: Comparision of predictions (argmax threshold)")
        preds = joint_probs.argmax(dim=-1, keepdim=False)
        predicts_the_same_class = (preds == preds[:, :1]).all(dim=1)
        print("\t\tProportion equal predictions among models:", 
              (predicts_the_same_class.sum() / predicts_the_same_class.shape[0]).round(decimals=2).item() * 100,
              "%")

