from copy import deepcopy
import os

from typing import Any, Dict, Tuple

import torch

from berebasl.simulation.credit_data_simulation import CreditDataGenerator, GaussianMixture

def slice_dgp_across_first_dim(dgp: CreditDataGenerator, i: int) -> CreditDataGenerator:
    kwargs_for_new = {
        vn : getattr(dgp, vn) for vn in [
                "prob_bad_given_no_shock",
                "prob_idiosyncratic_shock",
                "prob_bad_given_shock",
                "feats_noise_var"
            ]
    }
    mixture = dgp.good_mixture
    for mix_name in ("bad_mixture", "good_mixture"):
        mixture: GaussianMixture = getattr(dgp, mix_name)
        mu, cov_chol, weights = [t[i] if t is not None else t for t in mixture._normalized_params()]
        if weights is None:
            mu.squeeze_(0)
            cov_chol.squeeze_(0)

        kwargs_for_new[mix_name] = GaussianMixture(mean=mu, cov=cov_chol @ cov_chol.mT, weights=weights)

    return CreditDataGenerator(**kwargs_for_new)

def extract_objs_from_sim_dir(grid_path: str, sim_dir: str, device: torch.device = torch.device('cpu')):
    sim_results_path = os.path.join(grid_path, sim_dir, 'simulation_results_cv_mnar.pt')
    init_objs_path = os.path.join(grid_path, sim_dir, 'initial_simulation_objects_cv_mnar.pt')
    perf_bayes_stats_path = os.path.join(grid_path, sim_dir, "stats_perf_bayes.pt")
    perf_bayes_missing_stats_path = os.path.join(grid_path, sim_dir, "stats_perf_bayes_missing.pt")

    assert all([os.path.exists(p) for p in (sim_results_path, init_objs_path, perf_bayes_stats_path)])

    sim_results, init_objs, perf_bayes_stats, perf_bayes_missing_stats = [
        torch.load(p, map_location='cpu', weights_only=False)
        for p in (sim_results_path, init_objs_path, perf_bayes_stats_path, perf_bayes_missing_stats_path)
    ]

    data_generator: CreditDataGenerator = init_objs['data_generator'].to(device)
    stats = sim_results['stats']
    ## Extract the sample sizes as a tensor
    sample_sizes = torch.tensor(stats['sample_size'], device=device)
    ## stack the present lists of tensors
    stats = {
        k : (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)).to(device) 
        for k, v in stats.items()
    }
    perf_bayes_stats, perf_bayes_missing_stats = [{k: v.clone() for k, v in pbs.items()} for pbs in (perf_bayes_stats, perf_bayes_missing_stats)]

    corr_pos = sim_dir.find("_corr")
    bias_prop = float(sim_dir[len("bias_"):corr_pos].replace("_", "."))
    corr = float(sim_dir[corr_pos + len("_corr_"):].replace("_", "."))


    return {
            "Description" : f"$\\sigma_{{h, m}} = {corr}$, $\\mathbb{{P}}(bias) = {bias_prop}$",
            "corr_to_hidden" : corr,
            "bias_prop" : bias_prop,
            "stats" : stats,
            "perf_bayes_stats" : perf_bayes_stats,
            "perf_bayes_missing_stats" : perf_bayes_missing_stats,
            "alternative_accepted" : sim_results["alternative_accepted"],
            "sample_sizes" : sample_sizes,
            "credit_data" : sim_results["credit_data"].to(device),
            "data_generator" : data_generator,
            "k_folds" : init_objs["configs"]["k_folds"],
            "cv_count" : init_objs["configs"]["cv_count"],
            "var_to_hide" : init_objs["configs"]["var_to_hide"]
        }

def extract_objs_from_sim_dir_vec(
        grid_path: str,
        sim_dir: str,
        bias_prop: float,
        device: torch.device = torch.device('cpu')
    ):
    sim_results_path = os.path.join(grid_path, sim_dir, 'simulation_results_cv_mnar.pt')
    init_objs_path = os.path.join(grid_path, sim_dir, 'initial_simulation_objects_cv_mnar.pt')
    perf_bayes_stats_path = os.path.join(grid_path, sim_dir, "stats_perf_bayes.pt")
    perf_bayes_missing_stats_path = os.path.join(grid_path, sim_dir, "stats_perf_bayes_missing.pt")

    assert all([os.path.exists(p) for p in (sim_results_path, init_objs_path, perf_bayes_stats_path)])

    sim_results, init_objs, perf_bayes_stats, perf_bayes_missing_stats = [
        torch.load(p, map_location='cpu', weights_only=False)
        for p in (sim_results_path, init_objs_path, perf_bayes_stats_path, perf_bayes_missing_stats_path)
    ]

    data_generator: CreditDataGenerator = init_objs['data_generator'].to(device)
    stats = sim_results['stats']
    ## Extract the sample sizes as a tensor
    sample_sizes = torch.tensor(stats['sample_size'], device=device)
    ## stack the present lists of tensors
    stats = {
        k : (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)).to(device) 
        for k, v in stats.items()
    }
    perf_bayes_stats, perf_bayes_missing_stats = [{k: v.clone() for k, v in pbs.items()} for pbs in (perf_bayes_stats, perf_bayes_missing_stats)]


    return {
            "Description" : f"$\\mathbb{{P}}(bias) = {bias_prop}$",
            "corr_to_hidden" : data_generator.bad_mixture.cov[:, 0, init_objs["configs"]["var_to_hide"]],
            "bias_prop" : bias_prop,
            "stats" : stats,
            "perf_bayes_stats" : perf_bayes_stats,
            "perf_bayes_missing_stats" : perf_bayes_missing_stats,
            "alternative_accepted" : sim_results["alternative_accepted"],
            "sample_sizes" : sample_sizes,
            "credit_data" : sim_results["credit_data"].to(device),
            "data_generator" : data_generator,
            "k_folds" : init_objs["configs"]["k_folds"],
            "cv_count" : init_objs["configs"]["cv_count"],
            "var_to_hide" : init_objs["configs"]["var_to_hide"]
        }

def extract_all_sim_objs_vec(
        grid_path,
        biases: Dict[str, float],
        device: torch.device = torch.device('cpu')
        ) -> Dict[str, Dict[str, Any]]:
    all_sim_objs = {}
    for b_s, b_f in biases.items():
        all_sim_objs[b_s] = extract_objs_from_sim_dir_vec(
            grid_path,
            sim_dir='bias_' + b_s,
            bias_prop=b_f,
            device=device
        )
    return all_sim_objs

def extract_all_sim_objs(
        grid_path,
        biases: Dict[str, float],
        corrs: Dict[str, float]
        ) -> Dict[str, Dict[str, Any]]:
    all_sim_objs = {}
    for b_s in biases.keys():
        all_sim_objs[b_s] = {}
        for c_s in corrs.keys():
            all_sim_objs[b_s][c_s] = extract_objs_from_sim_dir(grid_path, f"bias_{b_s}_corr_{c_s}")
    return all_sim_objs

def extract_biases_and_corrs_from_dirnames(dir_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract bias and correlation dictionaries from directory names.
    
    Args:
        dir_path: Path to directory containing simulation folders
        
    Returns:
        Tuple of (biases_dict, corrs_dict) where keys are string representations
        and values are float values
    """
    biases = {}
    corrs = {}
    
    corr_ider = "_corr_"
    len_corr = len(corr_ider)
    bias_ider = "bias_"
    len_bias = len(bias_ider)
    
    for sim_name in os.listdir(dir_path):
        corr_pos = sim_name.find(corr_ider)
        if corr_pos == -1:
            continue
            
        b = sim_name[len_bias:corr_pos]
        if b not in biases:
            biases[b] = float(b.replace('_', '.'))
        
        c = sim_name[corr_pos + len_corr:]
        if c not in corrs:
            corrs[c] = float(c.replace('_', '.'))
    
    return biases, corrs

def extract_biases_from_dirnames(dir_path: str) -> Dict[str, float]:
    biases = {}
    
    bias_ider = "bias_"
    len_bias = len(bias_ider)
    
    for sim_name in os.listdir(dir_path):            
        b = sim_name[len_bias:]
        biases[b] = float(b.replace('_', '.'))

    biases = dict(sorted(biases.items(), key=lambda x: x[1]))
    
    return biases

def slice_vec_sim(sim: dict, wished_corr: float) -> dict:
    sim = {k : v.clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k,v in sim.items()}
    all_corrs: torch.Tensor = sim['corr_to_hidden']
    assert wished_corr in all_corrs
    idx_wished_cor = torch.where(all_corrs == wished_corr)[0].item()
    sim["corr_to_hidden"] = wished_corr

    ## Description changing
    orig_descr: str = sim["Description"] # $...$
    sim["Description"] = orig_descr[:-1] + rf", \sigma_{{mh}}={wished_corr:.2f}$"

    sim["stats"] = {
        k : (
            v if v.dim() == 1 else
            v[..., idx_wished_cor, :] if v.dim() == 4 else
            v[..., idx_wished_cor]
        )
        for k, v in sim['stats'].items()
    }

    for perf_b_key in ('perf_bayes_missing_stats', 'perf_bayes_stats'):
        sim[perf_b_key] = {
            k : v[idx_wished_cor * (v.size(0)//all_corrs.size(0))]
            for k, v in sim[perf_b_key].items()
        }

    credit_data = sim["credit_data"]
    for attr in ["features", "default_flag", "accepted"]:
        setattr(credit_data, attr, getattr(credit_data, attr)[idx_wished_cor])

    sim['alternative_accepted'] = {k : v[idx_wished_cor] for k, v in sim['alternative_accepted'].items()}

    sim["data_generator"] = slice_dgp_across_first_dim(sim["data_generator"], idx_wished_cor)

    return sim