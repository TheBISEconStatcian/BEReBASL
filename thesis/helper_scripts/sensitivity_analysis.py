from typing import Dict

import torch

from berebasl.simulation.credit_data_simulation import CreditDataGenerator

def extract_bayes_errors(
    all_sim_objs: dict,
    biases: Dict[str, float],
    corrs: Dict[str, float]
) -> torch.Tensor:
    bayes_errors = []

    B = len(biases)
    Co = len(corrs)
    
    for b_s in biases.keys():
        for c_s in corrs.keys():
            dgp: CreditDataGenerator = all_sim_objs[b_s][c_s]["data_generator"]
            bayes_errors.append(dgp.bayes_error_under_equal_covs())

    bayes_errors = torch.stack(bayes_errors, dim=0)
    bayes_errors = bayes_errors.reshape(B, Co, *bayes_errors.shape[1:])
    
    return bayes_errors