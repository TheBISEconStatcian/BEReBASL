import os

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
            "cv_count" : init_objs["configs"]["cv_count"]
        }