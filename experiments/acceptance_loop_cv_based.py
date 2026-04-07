"""
CV-based acceptance-feedback simulation.

Compared to the original acceptance_loop.py this script:

  1. Uses a CV rule (k-fold, repeated cv_count times) to derive decision thresholds
     under two criteria: KS-optimal and ROC-optimal (minimising Euclidean distance
     to (0,1) on the ROC curve).

  2. For every (decision_type × threshold_method × cv-trial + pooled-mean-trial)
     combination it records the *realised* statistical performance on the freshly
     drawn applicant batch, so we can compare what CV expected to what was actually
     observed and thereby detect selection bias.

  3. Three decision types are evaluated in parallel:
       - "acc_based"          : trained only on the accepted / observed sample
       - "oracle_naive"       : trained on the full unbiased population (ignoring
                                the accept flag) — naive oracle
       - "oracle_comparable"  : trained on accepted *and* rejected data but validated
                                only on the accepted fold — the "realistic oracle"
                                that accounts for the rejection structure

  k_folds and cv_count are intentionally hard-coded (5 / 10) so that the resume
  helper does not need to inspect them.

Usage
-----
  # fresh run
  python acceptance_loop_cv_based.py /path/to/output --create-path-if-missing

  # resume
  python acceptance_loop_cv_based.py /path/to/output --resume
  python acceptance_loop_cv_based.py /path/to/output --resume --num-gens 500
"""

from collections import defaultdict
import os
import time
from datetime import datetime
from warnings import warn

import torch

from typing import Any, Dict, List, Union, Tuple

from berebasl.simulation.acceptance_loop import (
    build_parser_for_loop,
    default_dgp,
    generate_initial_and_holdout_population,
    base_process_loop_args,
)
from berebasl.simulation.credit_data_simulation import (
    CreditData,
    CreditDataGenerator,
    CreditDataSample,
)
from berebasl.estimation.classifiers import BatchedLogistic
from berebasl.evaluation.k_fold_validation import (
    k_fold_cv_normalized_split,
    train_folds_from_cv_folds,
)
from berebasl.evaluation.batched_metrics import (
    batched_auroc_from_roc_points,
    batched_ks_statistic,
    batched_roc_points,
    optimal_roc_thresholds_from_roc_points,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (hard-coded so resume logic never needs to read them from a file)
# ──────────────────────────────────────────────────────────────────────────────

K_FOLDS: int = 5
CV_COUNT: int = 10
MIN_PER_FOLD: int = 128

# ──────────────────────────────────────────────────────────────────────────────
# ARG PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_args_cv_loop(args) -> dict:
    """
    Parse CLI args for the CV-based loop.

    We call base_process_loop_args but patch around the fact that it expects
    args.top_percent (used by the original loop but irrelevant here).  We inject
    a dummy value so the shared helper does not crash, then remove it from the
    result dict.
    """
    if not hasattr(args, "top_percent"):
        args.top_percent = 0.2          # dummy — not used by this script

    parsed = base_process_loop_args(args)
    parsed.pop("top_percent", None)     # CV loop does not use a top-percent rule
    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _pad_to_k_folds(tensor: torch.Tensor, k_folds: int) -> torch.Tensor:
    """Right-pad the last dimension of *tensor* with NaN up to *k_folds*."""
    missing = k_folds - tensor.size(-1)
    if missing == 0:
        return tensor
    if missing < 0:
        raise RuntimeError(
            "tensor last dim exceeds k_folds — k_folds may have been used "
            "inconsistently"
        )
    return torch.nn.functional.pad(
        tensor, pad=(0, missing), mode="constant", value=float("nan")
    )


def _repeat_cv(tensor: torch.Tensor, cv_count: int) -> torch.Tensor:
    """Prepend a new leading dimension of size *cv_count* by repeating."""
    trailing = tensor.dim()
    return tensor.unsqueeze(0).repeat(cv_count, *([1] * trailing))


# ──────────────────────────────────────────────────────────────────────────────
# CV CORE
# ──────────────────────────────────────────────────────────────────────────────

def _fit_logit_val_scores(
    train_feats: torch.Tensor,
    train_lbls: torch.Tensor,
    train_mask: torch.Tensor,
    val_feats: torch.Tensor,
) -> torch.Tensor:
    """Fit a batched logistic on train folds, return validation-fold scores."""
    lr = BatchedLogistic(
        n_features=train_feats.size(-1),
        batch_shape=train_lbls.shape[:-1],
        device=train_feats.device,
        dtype=train_feats.dtype,
    )
    lr.fit(train_feats, train_lbls, train_mask)
    return lr.predict_proba(val_feats)[..., -1]


def _batched_evaluation(
    scores_cv: torch.Tensor,
    lbls_cv: torch.Tensor,
    mask_cv: torch.Tensor,
    return_thresholds: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute KS and AUC (+ optionally their optimal thresholds) over batched
    CV score/label arrays.

    Parameters
    ----------
    scores_cv, lbls_cv, mask_cv :
        Tensors whose last dimension is the observation axis.
    return_thresholds :
        When True (default) also return the KS- and ROC-optimal thresholds.
    """
    dim = -1

    ks_vals, ks_thr = batched_ks_statistic(
        scores_cv, lbls_cv, mask_cv, dim, return_thresholds=True
    )

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
        "ks_stats":      ks_vals,
        "ks_thresholds": ks_thr,
        "roc_stats":     aucs,
        "roc_thresholds": roc_thr,
    }


def cross_validate(
    feats: torch.Tensor,
    lbls: torch.Tensor,
    k_folds: int,
    min_per_fold: int,
    cv_count: int,
    rng: torch.Generator = None,
) -> Dict[str, torch.Tensor]:
    """
    Standard k-fold CV on *feats* / *lbls* repeated *cv_count* times.

    Returns a dict of evaluation tensors, each padded to *k_folds* on the last
    dimension so that all CV runs have the same shape regardless of effective
    fold count.
    """
    N = lbls.size(-1)
    effective_k = min(N // min_per_fold, k_folds)

    feats_cv, lbls_cv, mask_cv = k_fold_cv_normalized_split(
        _repeat_cv(feats, cv_count),
        _repeat_cv(lbls, cv_count),
        rng=rng,
        k=effective_k,
        nan_lbls=float("nan") if torch.is_floating_point(lbls) else -1,
    )

    train_feats, train_lbls, train_mask = train_folds_from_cv_folds(
        feats_cv, lbls_cv, mask_cv, make_contiguous=True
    )
    scores = _fit_logit_val_scores(train_feats, train_lbls, train_mask, feats_cv)
    evals = _batched_evaluation(scores, lbls_cv, mask_cv)

    return {k: _pad_to_k_folds(v, k_folds) for k, v in evals.items()}


def realistic_oracle_cv(
    unb_feats: torch.Tensor,
    unb_lbls: torch.Tensor,
    acc_flag: torch.Tensor,
    k_folds: int,
    min_per_fold: int,
    cv_count: int,
    rng: torch.Generator = None,
) -> Dict[str, torch.Tensor]:
    """
    "Realistic oracle" CV: trains on *all* data (accepted + rejected) but
    evaluates only on the accepted validation fold.

    This mirrors the selection structure of the accepts-based classifier while
    still having access to the full dataset during training.
    """
    rej_flag = ~acc_flag

    # ── accepted split ────────────────────────────────────────────────────────
    feats_acc = _repeat_cv(unb_feats[acc_flag], cv_count)
    lbls_acc  = _repeat_cv(unb_lbls[acc_flag],  cv_count)
    N_acc     = lbls_acc.size(-1)
    effective_k = min(N_acc // min_per_fold, k_folds)

    feats_acc_cv, lbls_acc_cv, mask_acc_cv = k_fold_cv_normalized_split(
        feats_acc, lbls_acc, rng=rng, k=effective_k
    )

    # ── rejected split (same fold structure) ─────────────────────────────────
    feats_rej = _repeat_cv(unb_feats[rej_flag], cv_count)
    lbls_rej  = _repeat_cv(unb_lbls[rej_flag],  cv_count)
    feats_rej_cv, lbls_rej_cv, mask_rej_cv = k_fold_cv_normalized_split(
        feats_rej, lbls_rej, rng=rng, k=effective_k
    )

    cat_dim = lbls_rej_cv.dim() - 1
    train_feats, train_lbls, train_mask = [
        torch.cat([acc, rej], dim=cat_dim).contiguous()
        for acc, rej in zip(
            train_folds_from_cv_folds(feats_acc_cv, lbls_acc_cv, mask_acc_cv),
            train_folds_from_cv_folds(feats_rej_cv, lbls_rej_cv, mask_rej_cv),
        )
    ]

    # ── validate only on accepted fold ───────────────────────────────────────
    scores_acc_cv = _fit_logit_val_scores(
        train_feats, train_lbls, train_mask, feats_acc_cv
    )
    evals = _batched_evaluation(scores_acc_cv, lbls_acc_cv, mask_acc_cv)

    return {k: _pad_to_k_folds(v, k_folds) for k, v in evals.items()}


# ──────────────────────────────────────────────────────────────────────────────
# INIT / CHECKPOINT HELPERS  (CV-specific, not shared with original loop)
# ──────────────────────────────────────────────────────────────────────────────

def _results_path(sim_dir_path: str) -> str:
    return os.path.join(sim_dir_path, "simulation_results_cv.pt")


def _init_path(sim_dir_path: str) -> str:
    return os.path.join(sim_dir_path, "initial_simulation_objects_cv.pt")


def _check_and_save_init_cv_loop(
    sim_dir_path: str,
    data_generator: CreditDataGenerator,
    credit_data: CreditData,
    holdout_data: CreditData,
    base_seed: int,
    sample_size: int,
    num_gens: int,
    report_every: int,
    save_to_disc_every: int,
    persist_classifiers: bool,
    current_gen: int,
) -> str:
    """
    Validate arguments and persist initial simulation objects on the first run.

    Returns the path where rolling checkpoints will be saved.

    Differences from the original check_and_save_init_loop:
      - No classifier / BASL arguments (not needed here)
      - Saves to *_cv.pt files to avoid collisions with the original loop
    """
    if num_gens < 1:
        raise ValueError("num_gens must be ≥ 1")
    if not (1 <= current_gen <= num_gens):
        raise ValueError(f"current_gen must be in [1, num_gens={num_gens}]")

    init_p = _init_path(sim_dir_path)
    if os.path.exists(init_p) and current_gen == 1:
        raise AssertionError(
            f"current_gen=1 but {os.path.basename(init_p)} already exists "
            f"in {sim_dir_path}.  Use --resume to continue an existing run."
        )

    if not os.path.exists(init_p):
        torch.save(
            {
                "data_generator": data_generator,
                "initial_sample": credit_data,
                "holdout_data":   holdout_data,
                "configs": {
                    "base_seed":         base_seed,
                    "sample_size":       sample_size,
                    "num_gens":          num_gens,
                    "report_every":      report_every,
                    "save_to_disc_every": save_to_disc_every,
                    "persist_classifiers": persist_classifiers,
                    # Hard-coded CV params stored for documentation purposes only
                    "k_folds":    K_FOLDS,
                    "cv_count":   CV_COUNT,
                    "min_per_fold": MIN_PER_FOLD,
                },
            },
            f=init_p,
        )
        print(f"[INFO] Initial simulation objects saved to {init_p}")

    return _results_path(sim_dir_path)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def acceptance_loop(
    sim_dir_path: str,
    data_generator: CreditDataGenerator,
    credit_data: CreditData,
    holdout_data: CreditData,
    base_seed: int = 1807,
    sample_size: int = 256,
    num_gens: int = 300,
    report_every: int = 10,
    save_to_disc_every: int = 10,
    persist_classifiers: bool = True,
    current_gen: int = 1,
    stats: defaultdict = None,
) -> Tuple[CreditData, defaultdict]:
    """
    CV-based acceptance-feedback simulation loop.

    At each generation the loop:

      1. Runs k-fold CV (repeated *CV_COUNT* times) for three decision types:
           - acc_based        (observed / biased data)
           - oracle_naive     (unbiased data, naively)
           - oracle_comparable (realistic oracle)

      2. Derives accept/reject thresholds from KS- and ROC-optimal CV criteria.
         For each method we store:
           - the per-cv-trial threshold means  → rows 0 … CV_COUNT-1
           - the overall pooled mean threshold → row CV_COUNT   (most stable)

      3. Generates a new applicant batch, evaluates what each threshold would
         have *realised* as KS/AUC performance on that batch, and records it
         alongside the CV expectation.

      4. Accepts the new batch using the pooled-mean threshold of the
         acc_based / roc method (last decision type processed), adding them to
         credit_data for the next generation.

    Parameters
    ----------
    stats : defaultdict(list) or None
        Pre-populated stats dict when resuming; created fresh otherwise.
    """
    results_path = _check_and_save_init_cv_loop(
        sim_dir_path,
        data_generator,
        credit_data,
        holdout_data,
        base_seed,
        sample_size,
        num_gens,
        report_every,
        save_to_disc_every,
        persist_classifiers,
        current_gen,
    )

    if stats is None:
        stats = defaultdict(list)

    simulation_begin = time.time()
    times_needed: List[float] = []
    print("Checks passed — beginning CV-based acceptance loop")

    for gen_round_nr in range(current_gen, num_gens + 1):
        begin_round = time.time()

        # ── 0. Record dataset statistics ──────────────────────────────────────
        for k, val in credit_data.data_stats().items():
            stats[k].append(val)

        if gen_round_nr < num_gens:
            # ── 1. Collect data views ─────────────────────────────────────────
            current_sample: CreditDataSample = credit_data.to_sample_dataset()
            current_sample.manual_seed(base_seed + gen_round_nr + 1)

            unb_feats, unb_lbls, acc_flag = credit_data.unbiased_obs(
                include_accepted_status=True
            )

            # ── 2. CV for each decision type ──────────────────────────────────
            # We also keep fully-fitted classifiers for scoring the new batch.
            classifs: Dict[str, BatchedLogistic] = {}

            for c in ["acc_based", "oracle_naive", "oracle_comparable"]:
                if c == "oracle_comparable":
                    cv_results = realistic_oracle_cv(
                        unb_feats,
                        unb_lbls,
                        acc_flag,
                        K_FOLDS,
                        MIN_PER_FOLD,
                        CV_COUNT,
                        rng=current_sample.rng,
                    )
                else:
                    feats_for_cv = current_sample.features_labeled if c == "acc_based" else unb_feats
                    lbls_for_cv  = current_sample.labels  if c == "acc_based" else unb_lbls
                    
                    cv_results = cross_validate(
                        feats_for_cv,
                        lbls_for_cv,
                        K_FOLDS,
                        MIN_PER_FOLD,
                        CV_COUNT,
                        rng=current_sample.rng,
                    )

                # Store CV evaluation results
                for val_name, val in cv_results.items():
                    stats[c + "_" + val_name].append(val.detach().clone())

                # Fit a final classifier on all available data (not CV-split)
                # for use in scoring the incoming applicant batch.
                feats_full = current_sample.features_labeled if c == "acc_based" else unb_feats
                lbls_full  = current_sample.labels  if c == "acc_based" else unb_lbls
                clf = BatchedLogistic(
                    n_features=data_generator.features_count,
                    batch_shape=torch.Size([]),
                    device=feats_full.device,
                    dtype=feats_full.dtype,
                )
                clf.fit(feats_full, lbls_full)
                classifs[c] = clf

            # ── 3. Generate new applicant batch ───────────────────────────────
            data_generator.manual_seed(base_seed + gen_round_nr)
            feats_new, lbls_new = data_generator.sample(sample_size)   # [S,F], [S]

            # Expand labels for all (cv_count+1) threshold variants:
            #   rows 0…CV_COUNT-1 : per-cv-trial threshold mean
            #   row  CV_COUNT     : pooled mean (most stable estimate)
            exp_lbls = lbls_new.unsqueeze(0).expand(CV_COUNT + 1, -1)  # [M, S]

            # ── 4. Realised performance per (decision_type × method) ──────────
            # We track the variable that will be used for the actual acceptance
            # decision so we can set it correctly after the inner loops.
            final_accept_decision: torch.Tensor = None

            for dt in ["oracle_naive", "oracle_comparable", "acc_based"]:
                # Score new applicants with the fully-fitted classifier
                scores = classifs[dt].predict_proba(feats_new)[..., 1]  # [S]
                exp_scores = scores.unsqueeze(0).expand(CV_COUNT + 1, -1)  # [M, S]

                for m in ["ks", "roc"]:
                    # cv_thresholds : [CV_COUNT, K_FOLDS]
                    cv_thresholds: torch.Tensor = stats[dt + "_" + m + "_thresholds"][-1]
                    # Mean over folds for each CV trial → [CV_COUNT, 1]
                    cv_thr_means = cv_thresholds.nanmean(dim=-1, keepdim=True)
                    # Mean over CV trials → [1, 1]  (pooled / most-stable)
                    pooled_thr = cv_thr_means.nanmean(dim=0, keepdim=True)
                    # Stack: rows 0…CV_COUNT-1 are per-trial, row CV_COUNT is pooled
                    thresholds = torch.cat([cv_thr_means, pooled_thr], dim=0)  # [M, 1]

                    # Acceptance: score < threshold means the applicant is scored
                    # as *low risk* → accept.
                    accept_mask = exp_scores < thresholds  # [M, S]

                    # Realised performance on the new batch under each threshold
                    real_perf = _batched_evaluation(
                        exp_scores, exp_lbls, mask_cv=accept_mask,
                        return_thresholds=False
                    )
                    for s_n, s_v in real_perf.items():
                        if s_n.endswith("_stats"):
                            save_name = dt + "_" + m + "_real_" + s_n[:-6]
                            stats[save_name].append(s_v.detach().clone())

                    # Remember the acc_based / roc pooled decision for data ingestion
                    if dt == "acc_based" and m == "roc":
                        # Row CV_COUNT is the pooled (most stable) decision
                        final_accept_decision = accept_mask[CV_COUNT]  # [S]

            # ── 5. Add new generation using the pooled acc_based/roc decision ─
            if final_accept_decision is None:
                raise RuntimeError(
                    "final_accept_decision was never set — check loop ordering."
                )
            credit_data.add_gen(feats_new, lbls_new, accepted_new=final_accept_decision)

        # ── 6. Checkpoint ─────────────────────────────────────────────────────
        if (
            gen_round_nr == 1
            or gen_round_nr % save_to_disc_every == 0
            or gen_round_nr == num_gens
        ):
            torch.save(
                {
                    "credit_data":   credit_data,
                    "stats":         stats,
                    "gen_round_nr":  gen_round_nr,
                },
                results_path,
            )

        times_needed.append(time.time() - begin_round)
        if gen_round_nr % report_every == 0:
            print(
                "-- Finished Iteration",
                f"{gen_round_nr}/{num_gens}:",
                credit_data.count_accepts,
                "accepts and",
                credit_data.count_rejects,
                "rejects",
            )
            gen_rounds_left = num_gens - gen_round_nr
            times_tensor = torch.tensor(times_needed, device=credit_data.device, dtype=torch.float32)
            times_recorded_current = times_tensor.size(0)
            counts_per_round = credit_data.counts_per_round()
            times_matrix_expl = torch.stack([
                torch.ones_like(times_tensor),
                #counts_per_round["total"][-times_recorded_current:].cumsum(dim=0).to(times_tensor.dtype),
                torch.arange(current_gen, current_gen+times_recorded_current, dtype=times_tensor.dtype)
            ], dim=1)

            #print(times_matrix_expl)

            times_mat_inv = torch.linalg.inv(times_matrix_expl.mT.matmul(times_matrix_expl))

            betas_lin = times_mat_inv.matmul(times_matrix_expl.mT.matmul(times_tensor))
            betas_log = times_mat_inv.matmul(times_matrix_expl.mT.matmul(times_tensor.log()))

            avg_time = times_tensor.mean().item()
            data_until_end = gen_rounds_left*sample_size + credit_data.count_all
            future_times_expl = betas_lin.new_tensor([
                1,
                #data_until_end, 
                num_gens
            ])
            time_exp_lin = betas_lin.dot(future_times_expl).item()
            time_exp_log = betas_log.dot(future_times_expl).exp().item()

            print(
                f"\tRoughly expected time left: "
                f"{round(avg_time * gen_rounds_left / 60, 2)} min (AVG) or "
                f"{round(time_exp_lin / 60, 2)} min (lin, betas = {[round(b.item(),2) for b in betas_lin]}) or "
                f"{round(time_exp_log / 60, 2)} min (log, betas = {[round(b.item(),2) for b in betas_log]})"
            )

    print(
        "-- Simulation ended. Time needed:",
        round((time.time() - simulation_begin) / 60, 2),
        "min",
    )

    return credit_data, stats


# ──────────────────────────────────────────────────────────────────────────────
# RESUME HELPER
# ──────────────────────────────────────────────────────────────────────────────

def resume_cv_simulation_from_dir(
    sim_dir_path: str,
    new_gen_count: int = None,
) -> Tuple[CreditData, defaultdict]:
    """
    Resume a CV-based acceptance simulation from a saved directory.

    Looks for:
      - initial_simulation_objects_cv.pt
      - simulation_results_cv.pt

    Parameters
    ----------
    sim_dir_path : str
        Directory containing the above files.
    new_gen_count : int, optional
        If provided, override the stored num_gens (e.g. to extend a run).
    """
    print(f"\n[INFO] Attempting to resume CV simulation in: {sim_dir_path}")

    if not os.path.isdir(sim_dir_path):
        raise NotADirectoryError(
            f"Simulation directory does not exist: {sim_dir_path}"
        )

    init_p    = _init_path(sim_dir_path)
    results_p = _results_path(sim_dir_path)

    if not os.path.exists(init_p):
        raise FileNotFoundError(
            f"Missing {os.path.basename(init_p)} in {sim_dir_path}"
        )
    if not os.path.exists(results_p):
        raise FileNotFoundError(
            f"Missing {os.path.basename(results_p)} in {sim_dir_path}"
        )

    print("[INFO] Loading initial simulation objects…")
    init_objs = torch.load(init_p, map_location="cpu", weights_only=False)

    print("[INFO] Loading latest simulation results…")
    results = torch.load(results_p, map_location="cpu", weights_only=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"
        if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    configs: dict = init_objs["configs"]

    if new_gen_count is not None:
        old_num_gens = configs["num_gens"]
        if new_gen_count <= old_num_gens:
            warn(
                f"new_gen_count={new_gen_count} is not greater than the "
                f"current num_gens={old_num_gens}.  The run will stop at "
                f"{old_num_gens} unless new_gen_count > old_num_gens."
            )
        print(f"[INFO] Updating num_gens {old_num_gens} → {new_gen_count}")
        configs["num_gens"] = new_gen_count
        torch.save(init_objs, init_p)
        print("[INFO] Updated initial_simulation_objects_cv.pt saved.")

    # Move data objects to device
    data_generator: CreditDataGenerator = init_objs["data_generator"]
    holdout_data:   CreditData           = init_objs["holdout_data"]
    credit_data:    CreditData           = results["credit_data"]

    for obj in [data_generator, holdout_data, credit_data]:
        if hasattr(obj, "to"):
            obj.to(device)

    # Where to resume from
    current_gen: int = results["gen_round_nr"]

    # If the last checkpoint was at num_gens and we are not extending, nothing to do
    if current_gen >= configs["num_gens"]:
        raise RuntimeError(
            f"Simulation already completed at gen {current_gen} "
            f"(num_gens={configs['num_gens']}).  "
            "Pass new_gen_count > num_gens to extend."
        )

    print(
        f"[INFO] Resuming simulation from generation "
        f"{current_gen + 1}/{configs['num_gens']}"
    )

    stats: defaultdict = results["stats"]

    return acceptance_loop(
        sim_dir_path    = sim_dir_path,
        data_generator  = data_generator,
        credit_data     = credit_data,
        holdout_data    = holdout_data,
        base_seed       = configs["base_seed"],
        sample_size     = configs["sample_size"],
        num_gens        = configs["num_gens"],
        report_every    = configs["report_every"],
        save_to_disc_every = configs["save_to_disc_every"],
        persist_classifiers = configs["persist_classifiers"],
        current_gen     = current_gen + 1,   # start *after* last completed gen
        stats           = stats,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

# python -O -m experiments.acceptance_loop_cv_based --init-sample 1500  --sample-size 300

if __name__ == "__main__":
    argparser = build_parser_for_loop(
        desc=(
            "Run an acceptance-feedback simulation (Kozdoi et al. 2025) "
            "based on a CV rule and store results."
        )
    )
    # We deliberately do NOT add --top-percent here; process_args_cv_loop
    # injects a harmless dummy value so base_process_loop_args does not crash.

    args = argparser.parse_args()
    params = process_args_cv_loop(args)

    # ── Resume branch ─────────────────────────────────────────────────────────
    if params.pop("resume"):
        print("Resuming CV simulation…")
        resume_cv_simulation_from_dir(
            sim_dir_path  = params["sim_dir_path"],
            new_gen_count = params.get("num_gens"),
        )
        exit(0)

    # ── Fresh run ─────────────────────────────────────────────────────────────
    print("Begin of CV-based simulation.  Results to be saved in")
    print(params["sim_dir_path"])

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"
        if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        else "cpu"
    )

    torch.set_num_threads(24)
    torch.set_num_interop_threads(8)

    print("Simulation to be run on device:", device)
    print("Defining data generating process\n")

    data_generator: CreditDataGenerator = default_dgp(
        seed_credit_data_gen=params["initial_seed"],
        deterministic_weights_for_mixture_sampling=params.pop("deterministic_weights"),
        device=device,
        dtype=dtype,
    )

    print("Generating initial and holdout population")
    # Use top_percent=0.2 for the initial population rule (first-generation
    # heuristic accept/reject before CV scores are available).
    
    data_generator, credit_data, holdout_data = generate_initial_and_holdout_population(
        data_generator,
        params["initial_seed"],
        init_sample  = params.pop("init_sample"),   # default 1000 via --init-sample
        holdout_sample = params.pop("holdout_sample"),
        top_percent  = 0.2,
    )

    print("\n\n*** Starting CV-based acceptance loop ***\n\n")
    params["base_seed"] = params.pop("initial_seed")

    credit_data, stats = acceptance_loop(
        data_generator = data_generator,
        credit_data    = credit_data,
        holdout_data   = holdout_data,
        **params,
    )
