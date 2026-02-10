from sklearn.ensemble import IsolationForest
import numpy as np
import torch

from typing import Literal, Optional, Tuple, Union

from berebasl.estimation.bayesian_evaluation import BayesianMetric
from berebasl.estimation.classifiers import Classifier
from berebasl.simulation.credit_data_simulation import CreditDataSample

class BASLPartialUnbiaser:
    filtering_beta : torch.Tensor #filtering_beta in algorithm are both of these
    weak_learner : Classifier
    strong_learner : Classifier
    holdout_percent : float
    labeling_percent : float
    multiplier : float
    max_iterations :int
    early_stop : bool
    isolation_forest : IsolationForest

    def __init__(
            self,
            filtering_quantiles : dict[str, float], #filtering_beta in algorithm are both of these
            weak_learner : Classifier,
            strong_learner : Classifier,
            holdout_percent : float,
            sampling_percent : float,
            label_bads_percent : float, # labeling_percent in R implementation
            label_goods_percent : float, # labeling_percent / multiplier in R implementation
            max_iterations :int,
            early_stop : bool,
            isolation_forest : IsolationForest,
            bayesian_metric : BayesianMetric
    ):
        self.__class__.check_filtering_quantiles(filtering_quantiles)
        self.filtering_quantiles = filtering_quantiles

        for learner in [weak_learner, strong_learner]:
            if not Classifier.obj_has_needed_funs(learner):
                raise ValueError("weak_learner or strong_learner do not have methods predict_proba and fit as expected")
        
        self.weak_learner = weak_learner
        self.strong_learner = strong_learner
        self.holdout_percent = float(holdout_percent)
        self.sampling_percent = float(sampling_percent)
        self.label_bads_percent = float(label_bads_percent)
        self.label_goods_percent = float(label_goods_percent)

        members_expected_as_per = ["holdout_percent", "sampling_percent", "label_bads_percent", "label_goods_percent"]

        for member_var in members_expected_as_per:
            if not (0 <= getattr(self, member_var) <= 1):
                raise ValueError(f"{member_var} needs to be between 0 and 1")

        self.max_iterations = int(max_iterations)
        if self.max_iterations < 1:
            raise ValueError("max_iterations needs to be at least 1")
        
        self.early_stop = bool(early_stop)
        self.isolation_forest = isolation_forest
        self.bayesian_metric = bayesian_metric

        self.bayesian_metric.change_model(self.strong_learner)

    @staticmethod
    def check_filtering_quantiles(filtering_quantiles) -> None:
        if not 0.0 <= filtering_quantiles['lower'] <= 1.0:
            raise ValueError("filtering_quantiles['lower'] must be in the interval [0, 1].")
        if not 0.0 <= filtering_quantiles['upper'] <= 1.0:
            raise ValueError("filtering_quantiles['upper'] must be in the interval [0, 1].")
        if filtering_quantiles['lower'] >= filtering_quantiles['upper']:
            raise ValueError(
                "filtering_quantiles['lower'] must be strictly smaller than filtering_quantiles['upper']."
            )

    @property
    def should_filter(self):
        return (self.filtering_quantiles['lower'] > 0) or (self.filtering_quantiles['upper'] <1)
    
    def refit_model(
            self, 
            which_one : Literal["strong", "weak"], 
            features : torch.Tensor, 
            labels : torch.Tensor
    ) -> None:
        model = getattr(self, which_one + "_learner")
        model.fit(features, labels)

    def predict_proba_model(
            self, 
            which_one : Literal["strong", "weak"], 
            features : torch.Tensor,
            in_eval_if_possible : bool = False
    ) -> torch.Tensor:
        model = getattr(self, which_one + "_learner")
        if in_eval_if_possible and hasattr(model, "eval"):
            model.eval()
        return model.predict_proba(features)[..., 1]

    def self_learn(
            self,
            features_accept : torch.Tensor,
            default_flags_accept : torch.Tensor,
            features_reject : torch.Tensor,
            silent : bool = True
    ) -> torch.Tensor:
        pass

    def filter_rejects(
        self,
        features_rejects: torch.Tensor,
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
            features_rejects (torch.Tensor):
                Feature matrix of shape ``(n_samples, n_features)``.
            return_index (bool, optional):
                If ``True``, return a boolean mask indicating retained
                observations. If ``False``, return the filtered feature
                matrix. Defaults to ``True``.

        Returns:
            torch.Tensor:
                If ``return_index`` is ``True``, a boolean array of shape
                ``(n_samples,)`` indicating which observations are retained.
                Otherwise, a feature matrix containing only the retained
                observations.
        """
        
        features = features_rejects.detach().numpy()
        self.isolation_forest.fit(features)
        normality_scores = self.isolation_forest.score_samples(features)

        lower_score_bound, upper_score_bound = np.quantile(
            normality_scores,
            [self.filtering_quantiles["lower"], self.filtering_quantiles["upper"]],
        )

        keep_mask = (
                (lower_score_bound <= normality_scores)
                & (normality_scores <= upper_score_bound)
        )
        keep_mask = torch.from_numpy(keep_mask)

        return keep_mask if return_index else features_rejects[keep_mask]

    def bayesian_evaluation( #Next step to implement
            self,
            data : CreditDataSample,
            rejects_prior_probs : torch.Tensor,
            seed : Optional[int],
    ) -> torch.Tensor:
        if seed is not None:
            self.bayesian_metric.manual_seed(seed)

        return self.bayesian_metric.evaluate(data, rejects_prior_probs)
    
    def evaluate_labeled_performance(self, data : CreditDataSample, holdout_data : CreditDataSample) -> torch.Tensor:
        self.refit_model("strong", data.features_labeled, data.labels) # also refits the model in bayesian_metric
        strong_prob_bad_rejects = self.predict_proba_model(
            "strong", holdout_data.features_unlabeled, in_eval_if_possible=True
        )

        return self.bayesian_evaluation(
            holdout_data,
            rejects_prior_probs=strong_prob_bad_rejects,
            seed=1807
        )
    
    @staticmethod
    def modify_conf_preds_to_keep_max_labeling_bound(
            mask_conf_preds : torch.Tensor,
            upper_bound : int
    ) -> torch.Tensor:
        # 1) Normalize to shape [B, N]
        orig_shape = mask_conf_preds.shape
        should_flatten_dims = mask_conf_preds.dim() > 2
        if should_flatten_dims:
            mask_conf_preds = mask_conf_preds.flatten(0, -2)  # [B, N]

        orig_was_1d = mask_conf_preds.dim() == 1
        if orig_was_1d:
            mask_conf_preds = mask_conf_preds.unsqueeze(0)  # [1, N]

        B = mask_conf_preds.size(0)

        # Current number of trues per row
        count_conf_preds = mask_conf_preds.sum(dim=-1)  # [B]

        if (count_conf_preds > upper_bound).any():
            # Number of trues to KEEP per row: min(c_i, upper_bound)
            keep_counts = torch.clamp(count_conf_preds, max=upper_bound)  # [B]
            k_max = int(keep_counts.max().item())
            # Random scores, only finite in true positions of mask_conf_preds
            scores = torch.where(mask_conf_preds, torch.rand_like(mask_conf_preds, dtype=torch.float), -float('inf')) # [B, N]

            # Get up to k_max candidates per row
            _, topk_idx = scores.topk(k_max, dim=-1)  # [B, k_max]
            # Build per-row masks for how many to keep (some rows keep < k_max)
            idx_range = torch.arange(k_max, device=mask_conf_preds.device)  # [k_max]
            # valid_idx_mask[b, j] = j < keep_counts[b]
            # This mask already gets the idxs from topk_idx corresponding to the places where
            # mask_conf_preds should be true. This is ensured to contain a subset of **only** 
            # the positions where mask_conf_preds was already true, as top_k of scores is always
            # a position where mask_conf_preds was true
            valid_idx_mask = idx_range.unsqueeze(0) < keep_counts.unsqueeze(1)   # [B, k_max]

            # Initialize new mask all False
            new_mask = torch.zeros_like(mask_conf_preds, dtype=torch.bool)

            # Apply valid_idx_mask via advanced indexing
            # For rows where keep_counts[b] == 0, valid_idx_mask[b] is all False, so nothing is set
            batch_idx = torch.arange(B, device=mask_conf_preds.device).unsqueeze(1).expand_as(topk_idx)
            # Filter only positions to keep
            valid = valid_idx_mask
            new_mask[batch_idx[valid], topk_idx[valid]] = True

            mask_conf_preds = new_mask

        if orig_was_1d:
            mask_conf_preds = mask_conf_preds.squeeze(0)
        if should_flatten_dims:
            mask_conf_preds = mask_conf_preds.reshape(orig_shape)            

        return mask_conf_preds
    
    def confident_reject_labels(
            self, 
            data : CreditDataSample,
            hard_upper_bound_label_percent : bool = False, 
            rng : Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Refit weak learner with current labeled data
        self.refit_model("weak", data.features_labeled, data.labels)
        # 2. Gather indices for candidate rejects to label
        N, K = data.features_unlabeled.shape[-2:]
        normalized_shape = data.features_unlabeled.shape[:-1] # [..., N]
        device = data.features_unlabeled.device

        if self.sampling_percent == 1:
            M = N
            idx_candidate_rej_to_label = torch.arange(N, device=device).expand(*normalized_shape) # [..., M]
        else:
            M = round(self.sampling_percent * N)
            idx_shape = normalized_shape[:-1] + torch.Size([M])
            idx_candidate_rej_to_label = torch.rand(idx_shape, device=device).argsort()[..., :M] # [..., M]

        selected_unlabeled_features = data.features_unlabeled.gather(
            dim=-2, 
            index=idx_candidate_rej_to_label.unsqueeze(-1).expand( # # [..., M, K]
                *idx_candidate_rej_to_label.shape, K
            )
        )

        # 3. Calculate probability of default (bad)
        weak_prob_bad_rejects = self.predict_proba_model( # [..., M]
                "weak", 
                selected_unlabeled_features, 
                in_eval_if_possible=True
            )
        # 4. Define confidence levels required so that the desired percentage of labelling is kept
        conf_threshold_bad = weak_prob_bad_rejects.quantile(1-self.label_bads_percent, dim=-1, keepdim=True) # [..., 1]
        conf_threshold_good = 1-weak_prob_bad_rejects.quantile(self.label_goods_percent, dim=-1, keepdim=True).item() # [..., 1]

        # 5. Identify bad and good confident predictions
        lidx_conf_preds_bad = torch.zeros(normalized_shape, dtype=torch.bool, device=device) # [..., N]
        lidx_conf_preds_bad = lidx_conf_preds_bad.scatter( # [..., N]
            dim=-1,
            index=idx_candidate_rej_to_label,
            src= weak_prob_bad_rejects >= conf_threshold_bad # [..., M]
        )
        lidxs_conf_preds_good = torch.zeros_like(lidx_conf_preds_bad) # [..., N]
        lidxs_conf_preds_good = lidxs_conf_preds_good.scatter( # [..., N]
            dim=-1,
            index=idx_candidate_rej_to_label,
            src=weak_prob_bad_rejects <= (1-conf_threshold_good) # [..., M]
        )
        ## if wished enforce upper bound of labeling percent
        if hard_upper_bound_label_percent:
            lidxs_conf_preds_bad = self.__class__.modify_conf_preds_to_keep_max_labeling_bound(
                lidxs_conf_preds_bad, 
                upper_bound= round(M * self.label_bads_percent)
            ) # [..., N]
            lidxs_conf_preds_good = self.__class__.modify_conf_preds_to_keep_max_labeling_bound(
                lidxs_conf_preds_good, 
                upper_bound= round(M * self.label_goods_percent)
            ) # [..., N]

        lidx_conf_preds = lidxs_conf_preds_bad | lidxs_conf_preds_good # [..., N]

        # 6. Generate labels with dummy encoding in the dtype of the default_flag
        #    and filter the corresponding indices
        nan_val = torch.tensor(
            -1, 
            dtype=torch.int if data.labels.dtype == torch.bool else data.labels.dtype, 
            device=device
        )
        bad_val = torch.ones_like(nan_val)
        good_val = torch.zeros_like(nan_val)

        confident_preds = torch.where(
            lidxs_conf_preds_bad, 
            bad_val,
            torch.where(lidxs_conf_preds_good, good_val, nan_val)
        )

        mask_labeled_rejects = lidx_conf_preds

        return confident_preds, mask_labeled_rejects
    
    def basl_augment_sample(
            self, 
            data : CreditDataSample, 
            leave_orig_sample_untouched : bool = True, 
            early_stop : bool = True
        ) -> CreditDataSample:
        if early_stop:
            data, holdout_data = data.train_test_split(self.holdout_percent)
        
        if self.should_filter:
            data.features_unlabeled = self.filter_rejects(data.features_unlabeled, return_index=False)

        if early_stop:
            b_metric_last_iter = self.evaluate_labeled_performance(data, holdout_data)

        confident_preds, mask_infered = self.confident_reject_labels(data)
        if leave_orig_sample_untouched:
            data = data.label_rejects(inferred_labels=confident_preds, mask_inferred_rej_lbls=mask_infered, inplace = False)
        else:
            data.label_rejects(inferred_labels=confident_preds, mask_inferred_rej_lbls=mask_infered, inplace = True)

        for _ in range(self.max_iterations-1): # First iteration already done - ensure max iterations is kept
            next_iteration_is_sensible = mask_infered.any() and data.count_labeled > 0
            if not next_iteration_is_sensible:
                break

            if early_stop:
                b_metric_current_iter = self.evaluate_labeled_performance(data, holdout_data)
                reject_inference_stopped_improving_metric = b_metric_current_iter <= b_metric_last_iter
                if reject_inference_stopped_improving_metric:
                    break

                b_metric_last_iter = b_metric_current_iter

            # Next labeling stage
            confident_preds, mask_infered = self.confident_reject_labels(data)
            data.label_rejects(inferred_labels=confident_preds, mask_inferred_rej_lbls=mask_infered, inplace = True)
            # still need to find a way to keep track of which observations where labeled
            # and also a way to ensure that the percentage of obs to label is kept in hard way as in the r implementation

        return data


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