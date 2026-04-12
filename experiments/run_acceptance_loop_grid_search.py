"""
Grid-based experiments for CV-based acceptance simulation.

Runs a Cartesian product of noise_std and accept_flip_probability configurations,
each with fixed init-sample=1500 and sample-size=300, saving results in timestamped
subdirectories under berebasl/data/simulations.
"""

import os
import sys
from datetime import datetime
from itertools import product

import torch

# Add base path for imports
base_path = os.path.abspath("..")
if base_path not in sys.path:
    sys.path.append(base_path)

from .acceptance_loop_cv_based import build_parser_for_cv_loop, process_args_cv_loop, run_cv_simulation

def main():
    # Parse base arguments
    parser = build_parser_for_cv_loop()
    args = parser.parse_args([])
    
    # Process base params
    params = process_args_cv_loop(args)
    
    # Override fixed values
    params["init_sample"] = 1500
    params["sample_size"] = 300
    
    # Define grid
    noise_stds = [0.0, 0.2, 0.5, 0.8, 1.0]
    flip_probs = [0.0, 0.1, 0.2, 0.5, 0.8]
    
    # Create timestamped base directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_sim_dir = os.path.abspath(os.path.join("berebasl", "data", "simulations", f"grid_{timestamp}"))
    os.makedirs(base_sim_dir, exist_ok=True)
    
    print(f"Running grid experiments in {base_sim_dir}")
    
    # Run grid
    for noise_std, flip_prob in product(noise_stds, flip_probs):
        # Create subdirectory
        sub_dir_name = f"noise_{str(noise_std).replace('.', '_')}_flip_{str(flip_prob).replace('.', '_')}"
        sim_dir_path = os.path.join(base_sim_dir, sub_dir_name)
        os.makedirs(sim_dir_path, exist_ok=True)
        
        # Update params for this run
        run_params = params.copy()
        run_params["sim_dir_path"] = sim_dir_path
        run_params["noise_std"] = noise_std
        run_params["accept_flip_probability"] = flip_prob
        
        print(f"Starting experiment: noise_std={noise_std}, flip_prob={flip_prob} in {sim_dir_path}")
        
        # Run simulation
        try:
            run_cv_simulation(run_params, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64)
            print(f"Completed experiment: noise_std={noise_std}, flip_prob={flip_prob}")
        except Exception as e:
            print(f"Error in experiment noise_std={noise_std}, flip_prob={flip_prob}: {e}")
            continue

if __name__ == "__main__":
    main()