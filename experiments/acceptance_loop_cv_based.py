import os
import time

import torch

from typing import Any, Dict, List, Union, Tuple

from berebasl.simulation.acceptance_loop import (
    accept_based_on_top_percentent_of_arbitrary_var,
    build_parser_for_loop, 
    check_and_save_init_loop,
    default_dgp, 
    generate_initial_and_holdout_population,
    process_args_of_loop_parser
)
from berebasl.simulation.credit_data_simulation import CreditData, CreditDataGenerator, CreditDataSample
from berebasl.estimation.classifiers import Classifier, BatchedLogistic
from berebasl.evaluation.k_fold_validation import k_fold_cv_normalized_split
from berebasl.evaluation.batched_metrics import batched_ks_statistic

def cv_ks_vals_and_thresholds(
        feats: torch.Tensor,
        lbls: torch.Tensor,
        k_folds: int,
        min_per_fold: int,
        rng: torch.Generator = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, F = feats.shape
    effective_k_folds = min(N // min_per_fold, k_folds)

    feats_cv, lbls_cv, mask_cv = k_fold_cv_normalized_split(
        feats, 
        lbls, 
        rng=rng, 
        k=effective_k_folds,
        nan_lbls=float('nan') if torch.is_floating_point(lbls) else -1
    )

    train_feats, train_lbls, train_mask = train_folds_from_cv_folds(cv_feats, cv_lbls, cv_mask, make_contiguous=True)

    batched_lr = BatchedLogistic(
        n_features=F,
        batch_shape=train_lbls.shape[:-1],
        device=feats.device,
        dtype=feats.dtype
    )

    batched_lr.fit(train_feats, train_lbls, train_mask)

    scores_cv = batched_lr.predict_proba(feats_cv)[..., 1]
    cv_ks_values, cv_ks_thresholds = batched_ks_statistic(
        scores_cv,
        lbls_cv,
        mask_cv,
        dim=-1,
        return_thresholds=True
    )

    return cv_ks_values, cv_ks_thresholds

def acceptance_loop(
        sim_dir_path: str,
        data_generator: CreditDataGenerator,
        credit_data: CreditData,
        holdout_data: CreditData,
        base_seed: int = 1807,
        sample_size: int = 100,
        num_gens: int = 300,
        report_every: int = 10,
        save_to_disc_every: int = 10,
        persist_classifiers: bool = True,
        current_gen : int = 1,
        stats: List[Dict[str, Union[float, int]]] = [],
        classifiers_state_dicts: List[Dict[str, Union[Dict[str, Any], str]]] = []
) -> None:
    state_of_classifier_accepts, state_of_classifier_oracle, results_path = check_and_save_init_loop(
        sim_dir_path,
        data_generator,
        credit_data,
        holdout_data,
        classifier_accepts=None,
        classifier_oracle=None,
        basl_unbiaser=None,
        base_seed=base_seed,
        sample_size=sample_size,
        num_gens=num_gens,
        top_percent=0.2,
        report_every=report_every,
        save_to_disc_every=save_to_disc_every,
        persist_classifiers=persist_classifiers,
        current_gen=current_gen,
        stats=stats,
        classifiers_state_dicts=classifiers_state_dicts
    )

    simulation_begin = time.time()
    times_needed = []
    print("Checks passed in acceptance_loop, beginning now")

    k_folds = 5
    min_per_fold = 128

    for gen_round_nr in range(current_gen, num_gens + 1):          
        begin_round = time.time()
            
        ## Gather current statistics
        current_stats : dict = credit_data.data_stats()

        



        if gen_round_nr < num_gens:
            ## Get leakage-free data
            current_sample: CreditDataSample = credit_data.to_sample_dataset() # Ensure leakage-free data
            current_sample.manual_seed(base_seed + gen_round_nr + 1) # internal rng handles seeds for splitting

            unb_feats, unb_lbls, acc_flag = credit_data.unbiased_obs(include_accepted_status=True)

            # 1. Acc based
            current_stats["cv_ks_stat_acc_based"], cv_ks_thresholds = cv_ks_vals_and_thresholds(
                current_sample.feats, 
                current_sample.lbls, 
                k_folds, 
                min_per_fold,
                rng=current_sample.rng,
            )
            acceptance_threshold_biased = cv_ks_thresholds.mean().item() # [k_folds] -> float
            classifier_accepts = BatchedLogistic(
                n_features=data_generator.features_count,
                batch_shape=torch.Size([])
            )
            classifier_accepts.fit(current_sample.feats, current_sample.lbls)

            ## Generate new data
            data_generator.manual_seed(base_seed + gen_round_nr)
            # Let S:=sample_size
            feats_new_applicants, def_flag_new_applicants = data_generator.sample(sample_size) # [S, F], [S]
            with torch.no_grad():
                new_apps_prob_good_pred = classifier_accepts.predict_proba(feats_new_applicants)[..., 1] # [S]
                new_applicants_accepted = new_apps_prob_good_pred < acceptance_threshold_biased # [S]
                
            credit_data.add_gen(feats_new_applicants, def_flag_new_applicants, new_applicants_accepted)



        if gen_round_nr == 1 or gen_round_nr % save_to_disc_every == 0 or gen_round_nr == num_gens:
            classifiers_state_dicts.append({
                "gen_round_nr" : gen_round_nr,
                "classifiers" : {
                    "accepts_classifier" : state_of_classifier_accepts(),
                    "oracle_classifier" : state_of_classifier_oracle()
                }
            })

            checkpoint_to_save_to_disc = {
                "credit_data" : credit_data,
                "stats" : stats
            } | (
                {"classifiers_state_dicts" : classifiers_state_dicts} 
                if persist_classifiers else 
                {"gen_round_nr" : gen_round_nr}
            )

            torch.save(checkpoint_to_save_to_disc, results_path)

        times_needed.append(time.time() - begin_round)
        if gen_round_nr % report_every == 0:
            print("-- Finished Iteration", f"{gen_round_nr}/{num_gens}:", credit_data.count_accepts, 
                "accepts and", credit_data.count_rejects, " rejects")
            times_needed_tensor = torch.tensor(times_needed, dtype=torch.float32)
            gen_rounds_left = num_gens - gen_round_nr
            expected_time_left=times_needed_tensor.mean().item() * gen_rounds_left

            print("\tRoughly expected time left: ", round(expected_time_left/60, 2), "min")

    print("-- Simulation ended. Time needed:", round((time.time()-simulation_begin)/60, 2), "min")
    
    return credit_data, stats, classifiers_state_dicts

if __name__ == "__main__":
    argparser = build_parser_for_loop(
        desc="Run an acceptance-feedback simulation (Kozdoi et al. 2025) based on cv rule and store results."
    )

    params = process_args_of_loop_parser(argparser.parse_args())

