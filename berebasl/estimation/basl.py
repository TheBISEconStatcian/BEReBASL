from sklearn.ensemble import IsolationForest
import numpy as np
import torch

from typing import List, Literal, Optional, Tuple, Union

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

        self.isolation_forest = isolation_forest
        self.bayesian_metric = bayesian_metric

        self.bayesian_metric.change_model(self.strong_learner)

    def to_state_dict(self):
        members_to_take_as_plain = [
            "holdout_percent", 
            "sampling_percent", 
            "labels_bad_percent",
            "labels_good_percent",
            "max_iterations"
        ]
        state_dict = {k : v for k, v in self.__dict__ if k in members_to_take_as_plain}
        state_dict["filtering_quantiles"] = self.filtering_quantiles.copy() #this is a dictionary of floats

        state_dict["weak_learner_dict"] = self.weak_learner.to_state_dict()
        state_dict["strong_learner_dict"] = self.strong_learner.to_state_dict()

    @classmethod
    def from_state_dict(cls, state_dict : dict):
        kwargs_directly_passable = [
            "holdout_percent", 
            "sampling_percent", 
            "labels_bad_percent",
            "labels_good_percent",
            "max_iterations"
        ]
        kwargs_for_init = {k : v for k, v in state_dict.items() if k in kwargs_directly_passable}
        
        return cls(**kwargs_for_init)

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
        model.reset_parameters_to_initial()
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
    
    def _filter_single_batch_rejects(
            self,
            features : np.array
    ) -> np.ndarray:
        """
        Compute a two-sided Isolation Forest trimming mask for a single batch.

        This method fits the internal ``IsolationForest`` estimator on the
        provided feature matrix and computes normality scores via
        ``score_samples``. Observations are retained if their score lies within
        the central quantile interval defined by
        ``filtering_quantiles['lower']`` and ``filtering_quantiles['upper']``.
        Both highly anomalous and overly typical observations are removed.

        Parameters
        ----------
        features : np.ndarray
            A 2D NumPy array of shape ``(n_samples, n_features)`` containing
            the valid (non-padded) feature rows for a single batch.

        Returns
        -------
        np.ndarray
            A boolean array of shape ``(n_samples,)`` where ``True`` indicates
            that the observation is retained according to the two-sided
            trimming rule.
        """
        self.isolation_forest.fit(features)
        normality_scores = self.isolation_forest.score_samples(features) # [N]

        lower_score_bound, upper_score_bound = np.quantile(
            normality_scores,
            [self.filtering_quantiles["lower"], self.filtering_quantiles["upper"]],
        )

        keep_mask = (
                (lower_score_bound <= normality_scores)
                & (normality_scores <= upper_score_bound)
        )

        return keep_mask # [N]

    def filter_rejects(
        self,
        features_rejects: torch.Tensor,
        mask_valid_feats: torch.Tensor
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
                Tensor of shape ``(..., n_samples, n_features)`` containing
                matrices of shape ``(n_samples, n_features)``.
            mask_valid_feats (torch.Tensor):
                If ``True``, return a boolean mask indicating retained
                observations. If ``False``, return the filtered feature
                matrix. Defaults to ``True``.

        Returns:
            torch.Tensor:
                Boolean tensor of shape ``(..., n_samples)`` indicating which 
                observations are to be retained.

        Notes:
            The ``IsolationForest`` is a ``numpy`` implementation meaning that for each
            "super-batch" of the ``features_rejects`` is done sequentially and a cpu version
            of the ``mask_valid_feats`` and ``features_rejects`` is shortly created. The re-
            turn tensor gets transfered back to ``features_rejects.device`` at the end.
        """
        batch_shape = features_rejects.shape[:-2] 
        B = int(torch.prod(torch.tensor(batch_shape)))
        N = features_rejects.size(-2) 
        F = features_rejects.size(-1)
        features = features_rejects.reshape(B, N, F).detach().cpu()#.numpy()
        mask_valid = mask_valid_feats.reshape(B, N).detach().cpu()

        keep_masks = []

        for b in range(B):
            current_mask_valid = mask_valid[b]
            current_keep_mask = self._filter_single_batch_rejects(features[b][current_mask_valid].numpy()) # [B]
            current_keep_mask = torch.from_numpy(current_keep_mask)
            full_current_keep_mask = torch.zeros_like(current_mask_valid)
            full_current_keep_mask[current_mask_valid] = current_keep_mask

            keep_masks.append(full_current_keep_mask)

        keep_mask = torch.stack(keep_masks, dim=0).to(features_rejects.device).reshape(mask_valid_feats.shape)

        return keep_mask

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
            # Little abuse but ensures sampling the right percent
            # per batch and rng usage
            idx_candidate_rej_to_label, _ = data.generate_random_train_test_idxs(
                data.features_unlabeled.shape[:-1], 
                test_proportion=self.sampling_percent, 
                nan_mask = data.mask_nans_unlabeled
            )

        selected_unlabeled_features = data.features_unlabeled.gather(
            dim=-2, 
            index=idx_candidate_rej_to_label.unsqueeze(-1).expand( # [..., M, K]
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
        mask_conf_preds_bad = torch.zeros(normalized_shape, dtype=torch.bool, device=device) # [..., N]
        mask_conf_preds_bad = mask_conf_preds_bad.scatter( # [..., N]
            dim=-1,
            index=idx_candidate_rej_to_label,
            src= weak_prob_bad_rejects >= conf_threshold_bad # [..., M]
        )
        mask_conf_preds_good = torch.zeros_like(mask_conf_preds_bad) # [..., N]
        mask_conf_preds_good = mask_conf_preds_good.scatter( # [..., N]
            dim=-1,
            index=idx_candidate_rej_to_label,
            src=weak_prob_bad_rejects <= (1-conf_threshold_good) # [..., M]
        )
        ## if wished enforce upper bound of labeling percent
        if hard_upper_bound_label_percent:
            mask_conf_preds_bad = self.__class__.modify_conf_preds_to_keep_max_labeling_bound(
                mask_conf_preds_bad, 
                upper_bound= round(M * self.label_bads_percent)
            ) # [..., N]
            mask_conf_preds_good = self.__class__.modify_conf_preds_to_keep_max_labeling_bound(
                mask_conf_preds_good, 
                upper_bound= round(M * self.label_goods_percent)
            ) # [..., N]

        mask_conf_preds = mask_conf_preds_bad | mask_conf_preds_good # [..., N]

        # 6. Generate labels with dummy encoding in the dtype of the default_flag
        #    and filter the corresponding indices
        nan_val = data.labels_nan_value
        bad_val = torch.ones_like(nan_val)
        good_val = torch.zeros_like(nan_val)

        confident_preds = torch.where(
            mask_conf_preds_bad, 
            bad_val,
            torch.where(mask_conf_preds_good, good_val, nan_val)
        )

        mask_labeled_rejects = mask_conf_preds

        return confident_preds, mask_labeled_rejects
    
    def basl_augment_sample(
            self, 
            data : CreditDataSample, 
            leave_orig_sample_untouched : bool = True, 
            early_stop : bool = True
        ) -> CreditDataSample:
        if early_stop:
            data, holdout_data = data.train_test_split(self.holdout_percent)
        elif leave_orig_sample_untouched:
            data = data.clone()
        
        if self.should_filter:
            keep_mask = self.filter_rejects(
                features_rejects=data.features_unlabeled, 
                mask_valid_feats=~data.mask_nans_unlabeled
            )
            data.filter_unlabeled(keep_mask, inplace=True)

        if early_stop:
            b_metric_last_iter = self.evaluate_labeled_performance(data, holdout_data)

        confident_preds, mask_inferred = self.confident_reject_labels(data)
        data.label_rejects(inferred_labels=confident_preds, mask_inferred_rej_lbls=mask_inferred, inplace = True)

        for _ in range(self.max_iterations-1): # First iteration already done - ensure max iterations is kept
            next_iteration_is_sensible = mask_inferred.any() and data.count_labeled > 0
            if not next_iteration_is_sensible:
                break

            if early_stop:
                b_metric_current_iter = self.evaluate_labeled_performance(data, holdout_data)
                reject_inference_stopped_improving_metric = b_metric_current_iter <= b_metric_last_iter
                if reject_inference_stopped_improving_metric:
                    break

                b_metric_last_iter = b_metric_current_iter

            # Next labeling stage
            confident_preds, mask_inferred = self.confident_reject_labels(data)
            data.label_rejects(inferred_labels=confident_preds, mask_inferred_rej_lbls=mask_inferred, inplace = True)

        return data
