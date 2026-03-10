from typing import Optional, Callable

import torch

from berebasl.simulation.credit_data_simulation import CreditDataSample
from berebasl.estimation.classifiers import Classifier

class BayesianMetric:
    # Only implemented for binary classification right now
    # multiclass classification should be straight forward from here
    def __init__(
            self,
            model : Classifier,
            min_iterations : int,
            max_iterations : int,
            epsilon : float,
            metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            seed : Optional[int] = 1807,
            device : Optional[torch.device] = None
    ):
        if not Classifier.obj_has_needed_funs(model):
            raise ValueError("Model has not the fit and predict_proba functions with the expected signature")
        self.model = model
        self.min_iterations = int(min_iterations)
        self.max_iterations = int(max_iterations)
        self.epsilon = float(epsilon)
        self.metric = metric

        if device is None:
            device=torch.get_default_device()

        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

    def manual_seed(self, seed : int):
        self.rng.manual_seed(seed)

    def rng_to_device(self, device : torch.device, seed : Optional[int] = None, set_same_initial_seed : bool = True):
        if seed is None and set_same_initial_seed:
            seed = self.rng.initial_seed()

        self.rng = torch.Generator(device=device)

        if seed is not None:
            self.rng.manual_seed(seed)


    def predict_proba_model(self, features):
        return self.model.predict_proba(features)[..., 1]
    
    def change_model(
            self, 
            new_model: Classifier
        ) -> None:
        if not Classifier.obj_has_needed_funs(new_model):
            raise ValueError("Model has not the fit and predict_proba functions with the expected signature")
        
        self.model = new_model
    
    def sample_prior(self, prior_probs : torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(prior_probs, generator=self.rng)
    
    def evaluate(
            self, 
            data : CreditDataSample, 
            rejects_prior_probs : torch.Tensor
        ) -> torch.Tensor:
        """
        Note: The calculation of metrics_mean_absdiff bases on the fact that
        $$
        E_j - E_{j-1} = (M_j - E_{j-1})/j
        $$
        with $M_j$ the metric evaluated on the evaluation set with 
        """
        # data.features_accepts has shape [..., N, k] with k being the amount of features
        # just like data.features_rejects. Then data.default_flag_accepts.shape = [..., N]
        # so the same leading dimensions as the features
        
        preds_accept = self.predict_proba_model(data.features_labeled)
        preds_reject = self.predict_proba_model(data.features_unlabeled)

        joint_preds = torch.cat([preds_accept, preds_reject], dim=-1)

        
        metric_mean_up_to_last_it = joint_preds.new_full(size=data.labels.shape[:-1], fill_value=0)
        mask_nonconverged = data.labels.new_ones(size=data.labels.shape[:-1], dtype=bool)

        should_stop = False

        for it_nr in range(1, self.max_iterations + 1):
            reject_pseudo_labels = self.sample_prior(rejects_prior_probs)
            joint_labels = torch.cat([data.labels, reject_pseudo_labels], dim=-1)

            new_metric = self.metric(joint_preds, joint_labels)

            contribution_to_mean_of_new_metric = new_metric / it_nr

            should_check_convergence = it_nr >= self.min_iterations

            if should_check_convergence:
                metrics_mean_absdiff = (contribution_to_mean_of_new_metric - metric_mean_up_to_last_it/it_nr).abs()
                converged_in_current = metrics_mean_absdiff < self.epsilon

            metric_mean_up_to_last_it[mask_nonconverged] = (
                ((it_nr-1)/it_nr) * metric_mean_up_to_last_it[mask_nonconverged] + contribution_to_mean_of_new_metric[mask_nonconverged]
            )

            if should_check_convergence:
                mask_nonconverged = mask_nonconverged & ~converged_in_current
                should_stop = not mask_nonconverged.any()

            if should_stop:
                break

        return metric_mean_up_to_last_it
    


