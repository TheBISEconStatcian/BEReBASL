from copy import deepcopy

from typing import Literal, Dict

import torch

from berebasl.simulation.credit_data_simulation import CreditData

class ReplayBuffer:
    def __init__(
            self,
            credit_data: CreditData,
            alternative_accepted: Dict[str, torch.Tensor],
            wished_acc_base: Literal["roc", "ks"]
        ):
        self.credit_data = deepcopy(credit_data)
        key_wanted_acc = "acc_based_" + wished_acc_base
        if key_wanted_acc in alternative_accepted:
            self.credit_data.change_acceptance_flag(alternative_accepted[key_wanted_acc])