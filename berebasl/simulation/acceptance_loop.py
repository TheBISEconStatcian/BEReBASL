import argparse
import os
from datetime import datetime
from warnings import warn

from typing import Any, Dict, List, Union

from sklearn.ensemble import IsolationForest
import torch

from berebasl.estimation.basl import BASLPartialUnbiaser
from berebasl.estimation.bayesian_evaluation import BayesianMetric, batched_auroc
from berebasl.simulation.credit_data_simulation import CreditDataGenerator, CreditData, CreditDataSample
from berebasl.estimation.classifiers import Classifier, TorchLogistic

def build_parser_for_loop():
    parser = argparse.ArgumentParser(
        description="Run a simulation and store results."
    )

    # Optional positional argument
    parser.add_argument(
        "output_path",
        nargs="?",
        type=str,
        help="Directory where simulation results will be saved."
    )

    # Optional named argument
    parser.add_argument(
        "--output-path",
        dest="output_path_opt",
        type=str,
        help="Directory where simulation results will be saved (alternative to positional argument)."
    )

    parser.add_argument(
        "--create-path-if-missing",
        action="store_true",
        help="Create the output directory if it does not exist."
    )

    # Simulation parameters with defaults
    parser.add_argument("--initial-seed", type=int, default=1807)
    parser.add_argument("--init-sample", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--holdout-sample", type=int, default=3000)
    parser.add_argument("--num-gens", type=int, default=300)
    parser.add_argument("--top-percent", type=float, default=0.2)
    

    # Reporting and saving intervals
    parser.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="Print a progress report every N generations."
    )

    parser.add_argument(
        "--save-model-every",
        type=int,
        default=10,
        help="Save model snapshots every N generations."
    )

    return parser


def process_args_of_loop_parser(args):
    # Warn if both positional and optional paths are provided
    if args.output_path and args.output_path_opt:
        warn(
            "Both a positional path and --output-path were provided. "
            "Using the value from --output-path.",
            UserWarning
        )

    # Prefer the optional argument if provided
    path = args.output_path_opt or args.output_path

    if path:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise NotADirectoryError(
                    f"'{path}' exists but is not a directory."
                )
        else:
            if args.create_path_if_missing:
                os.makedirs(path, exist_ok=True)

            raise FileNotFoundError(
                f"Output path '{path}' does not exist. "
                "Use --create-path-if-missing to create it."
            )
    else:
        # No path provided → generate timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(
            os.path.join(__file__, f"../../data/simulations/simulation_{timestamp}")
        )

        os.makedirs(path, exist_ok=True)

     # Build the return dictionary 
    return { 
        "sim_dir_path": path, 
        "initial_seed": args.initial_seed, 
        "init_sample": args.init_sample, 
        "sample_size": args.sample_size, 
        "holdout_sample": args.holdout_sample, 
        "num_gens": args.num_gens, 
        "top_percent": args.top_percent, 
        "report_every": args.report_every, 
        "save_model_every": args.save_model_every, 
    }

def accept_based_on_top_percentent_of_arbitrary_var(
        features : torch.Tensor, 
        default_flag : torch.Tensor, 
        var_for_rule : int,
        top_percent : float,
        default_value : int = 1, # 1 or 0
        min_count_bads : int = 4
) -> torch.Tensor:
    if var_for_rule >= features.shape[1]:
        raise ValueError("var_for_rule outside of index")
    
    cutoff = torch.quantile(features[:, var_for_rule], 1 - top_percent)
    accepts = features[:, var_for_rule] >= cutoff

    count_defaults_within_accepts = (default_flag[accepts] == default_value).sum()
    if count_defaults_within_accepts < min_count_bads:
        mask_defaults_non_accepted = (~accepts) & (default_flag == default_value)
        defaults_still_selectable = mask_defaults_non_accepted.sum()
        if defaults_still_selectable == 0:
            return accepts
        
        count_bads_to_still_achieve = min_count_bads - count_defaults_within_accepts
        var_for_rule_vals_of_rejected_defaults = features[mask_defaults_non_accepted][:, var_for_rule]
        
        if defaults_still_selectable <= count_bads_to_still_achieve:
            var_for_rule_vals_of_rejected_defaults = features[mask_defaults_non_accepted][:, var_for_rule]
            accept_rule_to_include_all_defaults = features[:, var_for_rule] >=  var_for_rule_vals_of_rejected_defaults.min()
            return accept_rule_to_include_all_defaults
        
        new_cutoff = torch.topk(var_for_rule_vals_of_rejected_defaults,k=count_bads_to_still_achieve, largest=True).values[-1]

        return features[:, var_for_rule] >= new_cutoff
    
    return accepts

def acceptance_loop(
        sim_dir_path: str,
        data_gen: CreditDataGenerator,
        classifier_accepts: Classifier,
        classifier_oracle: Classifier,
        basl_unbiaser: BASLPartialUnbiaser,
        initial_seed: int = 1807,
        init_sample: int = 200,
        sample_size: int = 100,
        holdout_sample: int = 3000,
        num_gens: int = 300,
        top_percent: float = 200,
        report_every: int = 10,
        save_model_every: int = 10,
        determinstic_mixture_weights: bool = True
) -> None:
    # Initial population
    data_gen.manual_seed(initial_seed)
    feats_new_applicants, def_flag_new_applicants = data_gen.sample(
        init_sample, 
        determinstic_mixture_weights
    )
    
    accepts = accept_based_on_top_percentent_of_arbitrary_var(
        feats_new_applicants, 
        def_flag_new_applicants,
        var_for_rule=0,
        top_percent=top_percent,
        default_value = CreditDataGenerator.bad_good_encoding["bad"]
    )

    credit_data = CreditData(feats_new_applicants, def_flag_new_applicants, accepts)

    # Holdout Population
    data_gen.manual_seed(-initial_seed)
    holdout_features, holdout_flag = data_gen.sample(n=holdout_sample)
    holdout_data = CreditData(
        holdout_features, holdout_flag, 
        accepted_initial=torch.ones(holdout_flag.shape, dtype=torch.bool) # All are "accepted"
    )

    # Acceptance Loop
    ## Containers
    stats: List[Dict[str, Union[float, int]]] = []
    models_state_dicts: List[Dict[str, Union[Dict[str, Any], str]]] = []

    for gen_nr in range(1, num_gens + 1):
        if gen_nr % report_every == 0:
            print("-- Iteration", f"{gen_nr}/{num_gens}:", credit_data.accepted_count, 
                "accepts and", credit_data.rejected_count, " rejects")
            
        ## Gather current statistics
        current_stats : dict = credit_data.data_stats()
 
        ## Get leakage-free data
        current_sample: CreditDataSample = credit_data.to_sample_dataset() # Ensure leakage-free data
        current_sample.manual_seed(initial_seed + gen_nr + 1) # internal rng handles seeds for splitting

        ## Accepts based scorecard
        ### Reset params to ensure no effect of last calculation
        classifier_accepts.reset_parameters_to_initial()
        classifier_accepts.fit(current_sample.features_labeled, current_sample.labels)

        classifier_accepts.eval()
        holdout_probs_bad_accepts_based = classifier_accepts.predict_proba(holdout_data.features)[..., 1]
        current_stats["auc_accepts"] = batched_auroc(holdout_probs_bad_accepts_based, holdout_data.default_flag).item()

        ## Oracle scorecard
        classifier_oracle.reset_parameters_to_initial()
        classifier_oracle.fit(credit_data.features, credit_data.default_flag) # Using explicitly all data

        classifier_oracle.eval()
        holdout_probs_bad_oracle = classifier_oracle.predict_proba(holdout_data.features)[..., 1]
        current_stats["auc_biased"] = batched_auroc(holdout_probs_bad_oracle, holdout_data.default_flag).item()

        ## Corrected scorecard
        augmented_sample: CreditDataSample = basl_unbiaser.basl_augment_sample(
            data=current_sample, 
            leave_orig_sample_untouched=True, 
            early_stop=True
        )
        basl_unbiaser.refit_model(which_one='strong', features=augmented_sample.features_labeled, labels=augmented_sample.labels)
        holdout_probs_basl = basl_unbiaser.predict_proba_model(which_one='strong', features=holdout_data.features)
        current_stats["auc_basl"] = batched_auroc(holdout_probs_basl, holdout_data.default_flag).item()

        stats.append(current_stats)

        if gen_nr == 1 or gen_nr % save_model_every == 0 or gen_nr == num_gens:
            models_state_dicts.append({
                "gen_round" : gen_nr,
                "accepts_model" : classifier_accepts.to_state_dict(),
                "oracle_model" : classifier_oracle.to_state_dict(),
                "basl_strong_model" : basl_unbiaser.strong_learner.to_state_dict()
            })

        if gen_nr < num_gens:
            ## Generate new data
            data_gen.manual_seed(initial_seed + gen_nr)
            # Let S:=sample_size
            feats_new_applicants, def_flag_new_applicants = data_gen.sample(sample_size, determinstic_mixture_weights) # [S, F], [S]
            with torch.no_grad():
                new_applicants_pred_def_probs = classifier_accepts.predict_proba(feats_new_applicants)[..., 1] # [S]
                new_applicants_accepted = new_applicants_pred_def_probs >= new_applicants_pred_def_probs.quantile(1-top_percent) # [S]

            max_accepts_allowed = round(sample_size*top_percent)
            currently_accepted = new_applicants_accepted.sum()
            if currently_accepted > max_accepts_allowed:
                count_to_flip = currently_accepted - max_accepts_allowed
                idx_accepted = torch.where(new_applicants_accepted)[0] # [currently_accepted,] 
                perm = torch.randperm(currently_accepted, generator=data_gen.rng, device=data_gen.device) 
                idx_to_flip = idx_accepted[perm[:count_to_flip]]
                new_applicants_accepted[idx_to_flip] = False
                
            credit_data.add_gen(feats_new_applicants, def_flag_new_applicants, new_applicants_accepted)

if __name__ == "__main__":
    print(__file__)