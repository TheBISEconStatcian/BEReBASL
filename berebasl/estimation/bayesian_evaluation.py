from typing import Optional, Callable

import torch

from berebasl.simulation.credit_data_simulation import CreditDataSample

class BayesianMetric:
    # Only implemented for binary classification right now
    # multiclass classification should be straight forward from here
    def __init__(
            self,
            model,
            min_iterations : int,
            max_iterations : int,
            epsilon : float,
            metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            predict_model : Callable[["self.model", torch.Tensor], torch.Tensor] = lambda model, features_rejects : model.predict_proba(features_rejects)[..., 1],
            seed : Optional[int] = 1807,
            device : Optional[torch.device] = None
    ):
        self.model = model
        self.min_iterations = int(min_iterations)
        self.max_iterations = int(max_iterations)
        self.epsilon = float(epsilon)

        self.predict_model = predict_model
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


    def model_prediction(self, features_rejects):
        return self.predict_model(self.model, features_rejects)
    
    def update_model(
            self, 
            new_model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
            new_prediction_model: Optional[Callable[["self.model", torch.Tensor], torch.Tensor]] = None,
        ) -> None:
        if new_model is not None:
            self.model=new_model
        if new_prediction_model is not None:
            self.predict_model = new_prediction_model
    
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
        preds_accept = self.model_prediction(data.features_accepts)
        preds_reject = self.model_prediction(data.features_rejects)

        joint_preds = torch.cat([preds_accept, preds_reject], dim=-1)

        
        metric_mean_up_to_last_it = torch.full(size = data.default_flag_accepts.shape[:-1], fill_value=0.0)
        mask_nonconverged = torch.full(size = data.default_flag_accepts.shape[:-1], fill_value=True)

        should_stop = False

        for it_nr in range(1, self.max_iterations + 1):
            reject_pseudo_labels = self.sample_prior(rejects_prior_probs)
            joint_labels = torch.cat([data.default_flag_accepts, reject_pseudo_labels], dim=-1)

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