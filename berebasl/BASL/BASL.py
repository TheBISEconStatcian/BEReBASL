from sklearn.ensemble import IsolationForest
import numpy as np

import statsmodels.api as sm
from statsmodels.genmod import families

import torch

from typing import Union

def filter(
    ifo: IsolationForest,
    lower_trim_quantile: float,
    upper_trim_quantile: float,
    features_accept: np.ndarray,
    features_reject: np.ndarray,
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

    ifo.fit(features_accept)
    normality_ranking = ifo.score_samples(features_reject)

    lower_score_bound, upper_score_bound = np.quantile(
        normality_ranking,
        [lower_trim_quantile, upper_trim_quantile],
    )

    keep_mask = (
        (lower_score_bound <= normality_ranking)
        & (normality_ranking <= upper_score_bound)
    )

    return keep_mask if return_index else features_reject[keep_mask]

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

def accept_based_on_top_percentent_of_arbitrary_var(
        features : torch.Tensor, 
        default_flag : torch.Tensor, 
        var_for_rule : int,
        top_percent : float,
        default_value : int = 1, # 1 or 0
        min_count_bads : int = 4
):
    if var_for_rule >= features.shape[1]:
        raise ValueError("var_for_rule outside of index")
    
    cutoff = torch.quantile(features[:, var_for_rule], 1 - top_percent)

    accepts = features[:, var_for_rule] >= cutoff


    count_defaults_within_accepts = (default_flag[accepts] == default_value).sum()
    if count_defaults_within_accepts < min_count_bads:
        lidx_defaults_non_accepted = (~accepts) & (default_flag == default_value)
        defaults_still_selectable = lidx_defaults_non_accepted.sum()
        if defaults_still_selectable == 0:
            return accepts
        
        count_bads_to_still_achieve = min_count_bads - count_defaults_within_accepts
        var_for_rule_vals_of_rejected_defaults = features[lidx_defaults_non_accepted][:, var_for_rule]
        
        if defaults_still_selectable <= count_bads_to_still_achieve:
            var_for_rule_vals_of_rejected_defaults = features[lidx_defaults_non_accepted][:, var_for_rule]
            accept_rule_to_include_all_defaults = features[:, var_for_rule] >=  var_for_rule_vals_of_rejected_defaults.min()
            return accept_rule_to_include_all_defaults
        
        new_cutoff = torch.topk(var_for_rule_vals_of_rejected_defaults,k=count_bads_to_still_achieve, largest=True).values[-1]

        return features[:, var_for_rule] >= new_cutoff
    
    return accepts