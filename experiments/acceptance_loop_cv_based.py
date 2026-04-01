import torch

from berebasl.simulation.acceptance_loop import default_dgp, build_parser_for_loop, process_args_of_loop_parser

from berebasl.evaluation.k_fold_validation import k_fold_cv_normalized_split



if __name__ == "__main__":
    argparser = build_parser_for_loop(
        desc="Run an acceptance-feedback simulation (Kozdoi et al. 2025) based on cv rule and store results."
    )
    params = process_args_of_loop_parser(argparser.parse_args())