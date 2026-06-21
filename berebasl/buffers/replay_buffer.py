from copy import deepcopy

from typing import Literal, Dict

import torch

from berebasl.simulation.credit_data_simulation import CreditData

class CreditDataManager:
    def __init__(
            self,
            sim_objs: dict,
            wished_acc_base: Literal["roc", "ks"]
        ):
        self.credit_data = deepcopy(sim_objs["credit_data"])
        self.wished_acc_base = wished_acc_base
        key_wanted_acc = "acc_based_" + wished_acc_base
        alternative_accepted = sim_objs["alternative_accepted"]
        if key_wanted_acc in alternative_accepted:
            self.credit_data.change_acceptance_flag(alternative_accepted[key_wanted_acc])

        counts, th_meaning = simulation_counts_per_round(sim_objs)
        idx_wanted = [i for i, k in enumerate(th_meaning) if k == key_wanted_acc]
        assert len(idx_wanted) == 1
        idx_wanted = idx_wanted[0]

        self.counts_wished = counts[idx_wanted] # [AR, TBG, G]

    @property
    def N_acc(self):
        return self.credit_data.count_accepts
    
    @property
    def N_rej(self):
        return self.credit_data.count_rejects
    
    @property
    def N(self):
        return self.credit_data.count_all
    
    @property
    def F(self):
        return self.credit_data.features_count

    @property
    def mask_gens_with_defaults(self):
        return self.counts_wished[0, 1] > 0

    def plot_defaults_among_acc(self, title: str= '') -> None:
        accepted_defaults = self.counts_wished[0, 1, 1:].cpu()
        rounds = torch.arange(1, self.credit_data.last_gen_round+1, device=accepted_defaults.device)

        mask_was_zero = accepted_defaults==0

        plt.plot(rounds, accepted_defaults, label="Defaults count")
        plt.scatter(rounds[mask_was_zero], accepted_defaults[mask_was_zero], s=5, c='red', label="Points with Zero Defaults")

        plt.ylabel("Defaults among accepts")
        plt.xlabel("Generation round")

        plt.legend()
        plt.show()

    def get_round_ids_with_no_defaults_among_acc(self) -> torch.Tensor:
        accepted_defaults = self.counts_wished[0, 1]
        rounds = torch.arange(0, self.credit_data.last_gen_round+1, device=accepted_defaults.device)
        return rounds[accepted_defaults==0]

    def set_maximal_gen_round(self, gen_round: int):
        if not (0 <= gen_round <= self.credit_data.last_gen_round):
            raise AssertionError(f"gen_round has to be between 0 and self.credit_data.last_gen_round={self.credit_data.last_gen_round}")
        
        mask_keep = self.credit_data.gen_round <= gen_round

        attr_to_cut = ["features", "default_flag", "accepted", "gen_round"]
        for a in attr_to_cut:
            setattr(self.credit_data, a, getattr(self.credit_data, a)[mask_keep])

        self.credit_data.change_acceptance_flag(self.credit_data.accepted)
        self.counts_wished = self.counts_wished[..., :gen_round+1]