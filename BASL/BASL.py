from sklearn.ensemble import IsolationForest
import numpy as np

import statsmodels.api as sm
from statsmodels.genmod import families

def filter(
    ifo: IsolationForest,
    lower_trim_quantile: float,
    upper_trim_quantile: float,
    features: np.ndarray,
    return_index: bool = True,
) -> Union[np.ndarray, np.ndarray]:
    """
    Filters observations using a two-sided trimming strategy based on
    Isolation Forest normality scores.

    The function fits an Isolation Forest model on the provided feature
    matrix and computes the negative anomaly scores via
    ``IsolationForest.score_samples``. These scores induce a relative
    normality (similarity) ranking.

    Observations are retained if their score lies within the central
    quantile interval defined by ``lower_trim_quantile`` and
    ``upper_trim_quantile``. Consequently, both highly anomalous
    observations (lower tail) and overly typical observations
    (upper tail) are removed.

    This procedure corresponds to a two-sided percentile-based filtering
    scheme as described in Kozodoi et al. (2025), "Fighting Sampling Bias".

    Args:
        ifo (IsolationForest):
            An unfit ``IsolationForest`` instance used to compute
            normality scores.
        lower_trim_quantile (float):
            Lower quantile boundary in the interval ``[0, 1]``.
            Observations with scores below this quantile are discarded.
        upper_trim_quantile (float):
            Upper quantile boundary in the interval ``[0, 1]``.
            Observations with scores above this quantile are discarded.
        features (np.ndarray):
            Feature matrix of shape ``(n_samples, n_features)``.
        return_index (bool, optional):
            If ``True``, return a boolean mask indicating retained
            observations. If ``False``, return the filtered feature
            matrix. Defaults to ``True``.

    Returns:
        np.ndarray:
            If ``return_index`` is ``True``, a boolean array of shape
            ``(n_samples,)`` indicating which observations are retained.
            Otherwise, a feature matrix containing only the retained
            observations.

    Raises:
        ValueError:
            If ``lower_trim_quantile`` or ``upper_trim_quantile`` are
            outside the interval ``[0, 1]`` or if
            ``lower_trim_quantile >= upper_trim_quantile``.
    """
    if not 0.0 <= lower_trim_quantile <= 1.0:
        raise ValueError("lower_trim_quantile must be in the interval [0, 1].")
    if not 0.0 <= upper_trim_quantile <= 1.0:
        raise ValueError("upper_trim_quantile must be in the interval [0, 1].")
    if lower_trim_quantile >= upper_trim_quantile:
        raise ValueError(
            "lower_trim_quantile must be strictly smaller than upper_trim_quantile."
        )

    ifo.fit(features)
    normality_scores = ifo.score_samples(features)

    lower_score_bound, upper_score_bound = np.quantile(
        normality_scores,
        [lower_trim_quantile, upper_trim_quantile],
    )

    keep_mask = (
        (lower_score_bound <= normality_scores)
        & (normality_scores <= upper_score_bound)
    )

    return keep_mask if return_index else features[keep_mask]

def fit_and_predict_classic_logistic(X : np.array, y : np.array, add_intercept : bool = True):
    if add_intercept:
        X = np.column_stack([np.ones((X.shape[0],1), X.dtype), X])

    model = sm.GLM(
        endog=y,
        exog=X,
        family=families.Binomial # This specifies the logistic regression setup
    )
    fitted_model = model.fit()
    preds = fitted_model.predict()
    
    return preds