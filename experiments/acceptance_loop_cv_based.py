from collections import defaultdict
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
    base_process_loop_args
)
from berebasl.simulation.credit_data_simulation import CreditData, CreditDataGenerator, CreditDataSample
from berebasl.estimation.classifiers import BatchedLogistic
from berebasl.evaluation.k_fold_validation import k_fold_cv_normalized_split, train_folds_from_cv_folds
from berebasl.evaluation.batched_metrics import batched_auroc_from_roc_points, batched_ks_statistic, batched_roc_points, optimal_roc_thresholds_from_roc_points

# -------------------------
# FIXED ARG PROCESSING
# -------------------------
def process_args_of_loop_parser(args):
    base = base_process_loop_args(args)
    base["top_percent"] = args.top_percent if hasattr(args, "top_percent") else 0.2
    return base

# -------------------------
# UTILITIES
# -------------------------
def normalize_stat(to_normalize: torch.Tensor, k_folds: int) -> torch.Tensor:
    missing = k_folds - to_normalize.size(-1)

    if missing==0:
        return to_normalize
    if missing < 0:
        raise RuntimeError("to_normalize has too small of a last dim - k_folds was likely used in non-unified manner")
    
    normalized = torch.nn.functional.pad(
        to_normalize,
        pad=(0, missing),
        mode='constant',
        value=float('nan')
    )

    return normalized

def _repeat_for_cv_count(to_expand: torch.Tensor, cv_count: int) -> torch.Tensor:
    trailing_dims = to_expand.dim()
    return to_expand.unsqueeze(0).repeat(cv_count, *([1]*trailing_dims))

# -------------------------
# CV CORE
# -------------------------
def logit_val_scores(
        train_feats: torch.Tensor,
        train_lbls: torch.Tensor,
        train_mask: torch.Tensor,
        val_feats: torch.Tensor
) -> torch.Tensor:
    batched_lr = BatchedLogistic(
        n_features=train_feats.size(-1),
        batch_shape=train_lbls.shape[:-1],
        device=train_feats.device,
        dtype=train_feats.dtype
    )

    batched_lr.fit(train_feats, train_lbls, train_mask)

    val_scores = batched_lr.predict_proba(val_feats)[..., -1]
    return val_scores

def batched_evaluation(scores_cv, lbls_cv, mask_cv, return_thresholds=True):
    dim = -1

    ks_vals, ks_thr = batched_ks_statistic(scores_cv, lbls_cv, mask_cv, dim, return_thresholds=True)

    fpr, tpr, sorted_scores, _, sorted_mask = batched_roc_points(
        scores_cv, lbls_cv, mask_cv, dim
    )

    aucs = batched_auroc_from_roc_points(fpr, tpr, sorted_scores, sorted_mask, dim)

    roc_thr = None
    if return_thresholds:
        roc_thr = optimal_roc_thresholds_from_roc_points(
            fpr, tpr, sorted_scores, sorted_mask, dim
        )

    return {
        "ks_stats": ks_vals,
        "ks_thresholds": ks_thr,
        "roc_stats": aucs,
        "roc_thresholds": roc_thr
    }


def cross_validate(
        feats: torch.Tensor,
        lbls: torch.Tensor,
        k_folds: int,
        min_per_fold: int,
        cv_count: int,
        rng: torch.Generator = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = feats.size(-1)
    effective_k_folds = min(N // min_per_fold, k_folds)

    feats_cv, lbls_cv, mask_cv = k_fold_cv_normalized_split(
        _repeat_for_cv_count(feats, cv_count), 
        _repeat_for_cv_count(lbls, cv_count), 
        rng=rng, 
        k=effective_k_folds,
        nan_lbls=float('nan') if torch.is_floating_point(lbls) else -1
    )

    train_feats, train_lbls, train_mask = train_folds_from_cv_folds(
        feats_cv, lbls_cv, mask_cv, make_contiguous=True
    )   

    scores = logit_val_scores(train_feats, train_lbls, train_mask, feats_cv)
    evals = batched_evaluation(scores, lbls_cv, mask_cv)

    return {k: normalize_stat(v, k_folds) for k, v in evals.items()}

def realistic_oracle_cv(
        unb_feats: torch.Tensor,
        unb_lbls: torch.Tensor,
        acc_flag: torch.Tensor,
        k_folds: int,
        min_per_fold: int,
        cv_count: int,
        rng: torch.Generator = None
) -> torch.Tensor:
    # This function manages only partly batched input
    feats_acc = _repeat_for_cv_count(unb_feats[acc_flag], cv_count)
    lbls_acc = _repeat_for_cv_count(unb_lbls[acc_flag], cv_count)
    N_acc= lbls_acc.size(-1) # on N_acc we do the vali, that's why we take it
    effective_k_folds = min(N_acc // min_per_fold, k_folds)

    feats_acc_cv, lbls_acc_cv, mask_valid_acc_cv = k_fold_cv_normalized_split(
        feats_acc,
        lbls_acc,
        rng=rng,
        k=effective_k_folds
    )
    rej_flag = ~acc_flag
    feats_rej = _repeat_for_cv_count(unb_feats[rej_flag], cv_count)
    lbls_rej = _repeat_for_cv_count(unb_lbls[rej_flag], cv_count)
    feats_rej_cv, lbls_rej_cv, mask_valid_rej_cv = k_fold_cv_normalized_split(
        feats_rej,
        lbls_rej,
        rng=rng,
        k=effective_k_folds
    )
    cat_dim = lbls_rej_cv.dim()-1

    train_feats, train_lbls, train_mask = [
        torch.cat([acc, rej], dim=cat_dim).contiguous()
        for acc, rej in zip(
            train_folds_from_cv_folds(feats_acc_cv, lbls_acc_cv, mask_valid_acc_cv),
            train_folds_from_cv_folds(feats_rej_cv, lbls_rej_cv, mask_valid_rej_cv)
        )
    ]

    scores_acc_cv = logit_val_scores(
        train_feats, train_lbls, train_mask, feats_acc_cv
    )

    eval_results = batched_evaluation(scores_acc_cv, lbls_acc_cv, mask_valid_acc_cv)
    normalized_eval = {k : normalize_stat(v, k_folds) for k, v in eval_results.items()}

    return normalized_eval

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
        stats: defaultdict[list] = None,
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
    cv_count = 10

    if stats is None:
        stats = defaultdict(list)

    for gen_round_nr in range(current_gen, num_gens + 1):          
        begin_round = time.time()
            
        ## Gather current statistics
        for k, val in credit_data.data_stats().items():
            stats[k].append(val)


        if gen_round_nr < num_gens:
            # 0. Get data
            ## Get leakage-free data
            current_sample: CreditDataSample = credit_data.to_sample_dataset() # Ensure leakage-free data
            current_sample.manual_seed(base_seed + gen_round_nr + 1) # internal rng handles seeds for splitting
            ## Get unbiased data
            unb_feats, unb_lbls, acc_flag = credit_data.unbiased_obs(include_accepted_status=True)

            # Collect expectations and decissions
            classifs: Dict[str, BatchedLogistic] = {}
            for c in ["acc_based", "oracle_naive", "oracle_comparable"]:
                if c=="oracle_comparable":
                    cv_results = realistic_oracle_cv(
                        unb_feats, 
                        unb_lbls,
                        acc_flag,
                        k_folds, 
                        min_per_fold,
                        cv_count,
                        rng=current_sample.rng,
                    )
                else:
                    cv_results = cross_validate(
                        current_sample.feats if c=="acc_based" else unb_feats, 
                        current_sample.lbls if c=="acc_based" else unb_lbls, 
                        k_folds, 
                        min_per_fold,
                        cv_count,
                        rng=current_sample.rng,
                    )
                for val_name, val in cv_results.items():
                    stats[c + '_' + val_name].append(val.detach().clone())
                
                classifs[c] = BatchedLogistic(
                    n_features=data_generator.features_count,
                    batch_shape=torch.Size([])
                )
                classifs[c].fit(
                    current_sample.feats if c=="acc_based" else unb_feats, 
                    current_sample.lbls if c=="acc_based" else unb_lbls
                )

            # 2. Get new data

            ## Generate new data
            data_generator.manual_seed(base_seed + gen_round_nr)
            # Let S:=sample_size
            feats_new_applicants, lbls_new_applicants = data_generator.sample(sample_size) # [S, F], [S]
            exp_lbls_new_applicants = lbls_new_applicants.unsqueeze(0).repeat(cv_count+1, 1) # [cv+1, sample_size]

            decision_types = ["oracle_naive", "oracle_comparable", "acc_based"] # make sure acc_based is last
            methods = ["ks", "roc"]

            for dt in decision_types:
                scores = classifs[dt].predict_proba(feats_new_applicants)[..., 1]
                exp_scores = scores.expand(cv_count+1, -1) # [M, cv+1, sample_size]
                for m in methods:
                    cv_thresholds: torch.Tensor = stats[dt+'_'+m+'_thresholds'][-1] # [cv, k_folds]
                    cv_thr_means = cv_thresholds.nanmean(dim=-1, keepdim=True) # [cv, 1]
                    most_stable_thr_mean = cv_thr_means.mean(dim=0, keepdim=True) # [1, 1]
                    thresholds = torch.cat([cv_thr_means, most_stable_thr_mean], dim=0) # [cv + 1, 1]

                    accept_decision = exp_scores < thresholds # [cv+1, sample_size]

                    real_perf = batched_evaluation(exp_scores, exp_lbls_new_applicants, mask_cv=accept_decision, return_thresholds=False)
                    for s_n, s_v in real_perf.items():
                        if s_n[-6:] == "_stats":
                            save_name = dt + '_' + m + '_real'
                            stats[save_name].append(s_v.detach().clone())

                
            credit_data.add_gen(feats_new_applicants, lbls_new_applicants, accepted_new = accept_decision[-1, :])



        if gen_round_nr == 1 or gen_round_nr % save_to_disc_every == 0 or gen_round_nr == num_gens:
            checkpoint_to_save_to_disc = {
                "credit_data" : credit_data,
                "stats" : stats,
                "gen_round_nr" : gen_round_nr
            }

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
    
    return credit_data, stats

if __name__ == "__main__":
    argparser = build_parser_for_loop(
        desc="Run an acceptance-feedback simulation (Kozdoi et al. 2025) based on cv rule and store results."
    )
    argparser.add_argument("--top-percent", type=float, default=0.2)

    params = process_args_of_loop_parser(argparser.parse_args())

    # override defaults
    params["init_sample"] = 1000
    params["sample_size"] = 256

    dtype = torch.float64
    device = torch.device("cpu")

    data_gen = default_dgp(
        seed_credit_data_gen=params["initial_seed"],
        deterministic_weights_for_mixture_sampling=True,
        device=device,
        dtype=dtype
    )

    data_gen, credit_data, holdout_data = generate_initial_and_holdout_population(
        data_gen,
        params["initial_seed"],
        params["init_sample"],
        params["holdout_sample"],
        params["top_percent"]
    )

    acceptance_loop(
        data_generator=data_gen,
        credit_data=credit_data,
        holdout_data=holdout_data,
        base_seed=params["initial_seed"],
        sample_size=params["sample_size"],
        num_gens=params["num_gens"],
        sim_dir_path=params["sim_dir_path"]
    )