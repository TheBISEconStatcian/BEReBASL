import os

from typing import Any, Dict, Tuple

import torch

from berebasl.simulation.credit_data_simulation import CreditDataGenerator

def extract_objs_from_sim_dir(grid_path: str, sim_dir: str, device: torch.device = torch.device('cpu')):
    sim_results_path = os.path.join(grid_path, sim_dir, 'simulation_results_cv_mnar.pt')
    init_objs_path = os.path.join(grid_path, sim_dir, 'initial_simulation_objects_cv_mnar.pt')
    perf_bayes_stats_path = os.path.join(grid_path, sim_dir, "stats_perf_bayes.pt")

    assert all([os.path.exists(p) for p in (sim_results_path, init_objs_path, perf_bayes_stats_path)])
    
    sim_results = torch.load(sim_results_path, map_location='cpu', weights_only=False)
    init_objs = torch.load(init_objs_path, map_location='cpu', weights_only=False)
    perf_bayes_stats = torch.load(perf_bayes_stats_path, map_location='cpu', weights_only=False)

    data_generator: CreditDataGenerator = init_objs['data_generator'].to(device)
    stats = sim_results['stats']
    ## Extract the sample sizes as a tensor
    sample_sizes = torch.tensor(stats['sample_size'], device=device)
    ## stack the present lists of tensors
    stats = {
        k : (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)).to(device) 
        for k, v in stats.items()
    }
    perf_bayes_stats = {k: v.clone() for k, v in perf_bayes_stats.items()}

    corr_pos = sim_dir.find("_corr")
    bias_prop = float(sim_dir[len("bias_"):corr_pos].replace("_", "."))
    corr = float(sim_dir[corr_pos + len("_corr_"):].replace("_", "."))


    return {
            "Description" : f"$\\sigma_{{h, m}} = {corr}$, $\\mathbb{{P}}(bias) = {bias_prop}$",
            "corr_to_hidden" : corr,
            "bias_prop" : bias_prop,
            "stats" : stats,
            "perf_bayes_stats" : perf_bayes_stats,
            "alternative_accepted" : sim_results["alternative_accepted"],
            "sample_sizes" : sample_sizes,
            "credit_data" : sim_results["credit_data"].to(device),
            "data_generator" : data_generator,
            "k_folds" : init_objs["configs"]["k_folds"],
            "cv_count" : init_objs["configs"]["cv_count"],
            "var_to_hide" : init_objs["configs"]["var_to_hide"]
        }

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