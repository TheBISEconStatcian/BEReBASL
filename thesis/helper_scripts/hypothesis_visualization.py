from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from matplotlib import pyplot as plt

import numpy as np
import torch

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from berebasl.simulation.credit_data_simulation import CreditData
from berebasl.utils.normalized_shape_tensor_ops import masked_batched_trapz
from experiments.acceptance_loop_cv_based_MNAR import EXPECTATION_TYPES, METRIC_CATEGORIES, THRESHOLD_BASIS, REAL_PERFORMANCE_TYPES

# Calculate "Absolute" expectation and realization lines

def gen_lines_colors_and_main(
        stat: torch.Tensor, 
        sample_sizes: torch.Tensor, 
        base_color: Tuple[float],
        main_lbl: str, 
        alpha_dict: Dict[int, float], 
        lines: List[np.ndarray] = None, 
        colors: List[Tuple[float]] = None,
        sort_stat: bool = True
    )-> Tuple[List[np.ndarray], List[Tuple[float]], Dict[str, Union[List[np.ndarray], Dict[str, Any]]]]:
    stat_dim = stat.dim()
    if stat_dim == 0:
        raise ValueError("stat should have at least one dim")
    
    if lines is None:
        lines = []
    if colors is None:
        colors = []
    
    if stat_dim > 1:
        if sort_stat:
            stat = stat.sort(dim=0).values
        broadcasted_sizes = sample_sizes.expand_as(stat) # [k, cv, gen]

        stat_with_sizes = torch.stack([broadcasted_sizes, stat], dim=-1).flatten(0,-3)

        lines.extend(
            [s.detach().cpu().numpy() 
             for s in stat_with_sizes]
        )
        
        colors.extend([base_color + (alpha_dict.get(stat_dim, 0.5),)] * stat_with_sizes.size(0))

        return gen_lines_colors_and_main(
            stat.nanmean(dim=0),
            sample_sizes,
            base_color,
            main_lbl,
            alpha_dict,
            lines,
            colors
        )
    
    main_dict = {
            "args" : [t.detach().cpu().numpy() for t in (sample_sizes, stat)],
            "kwargs" : {
                "color" : base_color + (alpha_dict.get(stat_dim, 1),),
                "linewidth" : 1.2,
                "label" : main_lbl
            }
        }
    
    return lines, colors, main_dict

def linetypes_labels_and_colors_for_exp_real_plot(
        only_related_exp: bool = True,
        diff_plot: bool = False
):
    linetypes = ["exp", "exp_bayes"] + REAL_PERFORMANCE_TYPES + ["real_bayes"]
    labels_linetypes = {
        "exp" : "Expectation (logit)",
        "exp_bayes" : "Expectation (Perf. Bayes)",
        "biased_acc" : "Real. on Biased Accepts",
        "unbiased_acc" : "Real. on Unbiased Accepts",
        "unbiased_future" : "Real. on Unbiased Future Observations",
        "real_bayes" : "Real. Perf. Bayes on related Exp",
    }
    if only_related_exp:
        for k in ['biased_acc', 'unbiased_acc', 'unbiased_future']:
            labels_linetypes[k] = "Realized (logit)"
        labels_linetypes["real_bayes"] = "Realized (Bayes)"
        if diff_plot:
            labels_linetypes["exp"] = "Logit"
            labels_linetypes["exp_bayes"] = "Bayes"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_map = {}
    if only_related_exp:
        jump_points = ["exp"]
        if not diff_plot:
            jump_points += ["exp_bayes", "unbiased_future"]
        idx_colors = 0
        for e in linetypes:
            colors_map[e] = to_rgb(colors[idx_colors])
            if e in jump_points:
                idx_colors += 1
    else:
        colors_map = {e : to_rgb(c) for e, c in zip(linetypes, colors[:len(linetypes)])}

    return linetypes, labels_linetypes, colors_map

ALL_THRESHOLD_LBLS = [t+'_'+m for m in METRIC_CATEGORIES for t in THRESHOLD_BASIS]

def gather_all_acc_flags(sim_objs) -> Dict[str, torch.Tensor]:
    acc_flags: Dict[str, torch.Tensor] = {k : v.clone() for k, v in sim_objs['alternative_accepted'].items()}
    lbl_acc_flag_in_data = set(ALL_THRESHOLD_LBLS) - set(acc_flags.keys())
    assert len(lbl_acc_flag_in_data) == 1
    lbl_acc_flag_in_data = next(lbl_acc_flag_in_data.__iter__())

    credit_data: CreditData = sim_objs["credit_data"]
    acc_flags[lbl_acc_flag_in_data] = credit_data.accepted.clone()

    return acc_flags

def get_acc_defaults_counts_per_round(sim_objs) -> Dict[str, torch.Tensor]:
    acc_flags = gather_all_acc_flags(sim_objs)
    credit_data: CreditData = sim_objs["credit_data"]

    rounds = credit_data.gen_round
    G = credit_data.last_gen_round + 1 # Count rounds

    boolean_def_flag = credit_data.default_flag.to(bool)

    return {
        k : v.new_zeros((G,), dtype=torch.int32).scatter_add_(
            dim=-1,
            index=rounds,
            src=(v & boolean_def_flag).to(torch.int32)
        )
        for k, v in acc_flags.items()
    }

def simulation_counts_per_round(
        sim_objs,
        dtype_counts: torch.dtype = torch.int32
    ) -> Dict[str, torch.Tensor]:
    credit_data: CreditData = sim_objs["credit_data"]

    rounds = credit_data.gen_round
    N = rounds.size(0)
    G = credit_data.last_gen_round + 1 # Count rounds

    
    accepts_per_th = gather_all_acc_flags(sim_objs)
    threshold_types = [
        "through_the_door"
    ] + list(accepts_per_th.keys())
    TT = len(threshold_types)
    AR = 2 # acc reject

    th_flags = rounds.new_ones((TT, N), dtype=bool)

    for t_idx, a_t in enumerate(accepts_per_th.values()):
        th_flags[t_idx+1] = a_t

    th_ar_flags = th_flags.unsqueeze(1).repeat(1,2,1) # [TH, AR, N]
    th_ar_flags[:, 1] = ~th_ar_flags[:, 0]

    
    BG = 2
    bg_flags = th_ar_flags.new_empty((BG, N))
    bg_flags[0] = credit_data.default_flag.to(bool) # bad flag
    bg_flags[1] = ~bg_flags[0]

    count_base = (
        th_ar_flags.unsqueeze(2) & # [TT, AR, 1, N]
        bg_flags.view(1, 1, BG, N)
    ).to(dtype_counts) # [TT, AR, BG, N]

    TBG = 1 + BG

    counts = count_base.new_zeros((TT, AR, TBG, G))

    counts[:, :, 1:].scatter_add_(
        dim=-1,
        index = rounds.expand(TT, AR, BG, N),
        src = count_base
    )

    counts[:, :, 0] = counts[:, :, 1:].sum(dim=2)

    return counts, threshold_types # [TT, AR, TBG, G]

def generate_line_collections_for_exp_real_plot(
        sim_objs : Dict[str, torch.Tensor],
        only_related_exp: bool = True,
        add_real_bayes: bool = True,
        use_bayes_with_missing: bool = True
    ):
    linetypes, labels_linetypes, colors = linetypes_labels_and_colors_for_exp_real_plot(
        only_related_exp
    )

    alpha_dict_exp = {3 : 0.03, 2 : 0.4, 1 : 0.8}
    alpha_dict_real = {3: 0.1, 2: 0.6, 1 : 0.8}

    mask_no_bads_among_acc = {
        k : v == 0 for k, v in get_acc_defaults_counts_per_round(sim_objs).items()
    }

    stats = sim_objs["stats"]
    bayes_stats_lbl = "perf_bayes_"
    if use_bayes_with_missing:
        bayes_stats_lbl += "missing_"
    bayes_stats_lbl += "stats"
    bayes_stats = sim_objs[bayes_stats_lbl]
    sample_sizes = sim_objs["sample_sizes"]

    line_collections_data: Dict[str, Dict[str, List]] = {}

    for exp in EXPECTATION_TYPES:
        for m in METRIC_CATEGORIES:
            current_lbl = exp + '_' + m
            line_collections_data[current_lbl] = {
                "lines" : [],
                "colors" : [],
                "mains" : []
            }

            # 1. CV x K lines - plot all cv lines of the expectation
            expected_stat = stats[exp+'_'+m+'_stats'] # [gen, cv, k]

            lines_exp, colors_exp, main_exp = gen_lines_colors_and_main(
                stat=stats[exp+'_'+m+'_stats'].transpose(0,-1), # [k, cv, gen]
                sample_sizes=sample_sizes,
                base_color=colors["exp"],
                main_lbl=labels_linetypes["exp"],
                alpha_dict=alpha_dict_exp,
                #lines=[],
                #colors=[]
            )

            line_collections_data[current_lbl]["lines"].extend(lines_exp)
            line_collections_data[current_lbl]["colors"].extend(colors_exp)
            line_collections_data[current_lbl]["mains"].append(main_exp)

            if add_real_bayes:
                lines_exp_bayes, colors_exp_bayes, main_exp_bayes = gen_lines_colors_and_main(
                    stat=bayes_stats[exp+'_'+m+'_stats'][:-1].transpose(0,-1), # [k, cv, gen]
                    sample_sizes=sample_sizes,
                    base_color=colors["exp_bayes"],
                    main_lbl=labels_linetypes["exp_bayes"],
                    alpha_dict=alpha_dict_exp,
                    #lines=[],
                    #colors=[]
                )

                line_collections_data[current_lbl]["lines"].extend(lines_exp_bayes)
                line_collections_data[current_lbl]["colors"].extend(colors_exp_bayes)
                line_collections_data[current_lbl]["mains"].append(main_exp_bayes)

            th_is_oracle = exp.startswith("oracle")
            th = "oracle" if th_is_oracle else exp
            assert th in THRESHOLD_BASIS, "Threshold logic changed"

            for perf in REAL_PERFORMANCE_TYPES:
                perf_is_unb_future = perf=="unbiased_future"
                should_plot_perf = False
                if exp == 'acc_based':
                    should_plot_perf = perf == "biased_acc"
                elif exp == 'oracle_naive':
                    should_plot_perf = perf == "unbiased_future"
                elif exp=='oracle_comparable':
                    should_plot_perf = perf == 'unbiased_acc'
                if perf_is_unb_future:
                    real_lbl_stats = perf+ '_' + th
                    if not th_is_oracle:
                        # If not oracle, the available history also depend on the basis
                        # of the threshold metric and can/should generate different
                        # scores which of course **do** affect the realized performance
                        real_lbl_stats += "_" + m # Make sure based on same metric for acc-based
                    real_lbl_stats += '_real_' + m
                    realized = stats[real_lbl_stats] # [gen]

                    # lines_real and colors_real are empty lists
                    if not only_related_exp or should_plot_perf:
                        lines_real, colors_real, main_real = gen_lines_colors_and_main(
                            stat=realized.masked_fill(
                                mask_no_bads_among_acc[th+'_'+m][1:], float('nan')
                            ),
                            sample_sizes=sample_sizes,
                            base_color=colors[perf],
                            main_lbl=labels_linetypes[perf],
                            alpha_dict=alpha_dict_real # not used - passed "as dummy"
                        )
                    if should_plot_perf and add_real_bayes:
                        lines_real_bayes, colors_real_bayes, main_real_bayes = gen_lines_colors_and_main(
                            stat=bayes_stats[real_lbl_stats][1:].masked_fill(
                                mask_no_bads_among_acc[th+'_'+m][1:], float('nan')
                            ),
                            sample_sizes=sample_sizes,
                            base_color=colors["real_bayes"],
                            main_lbl=labels_linetypes["real_bayes"],
                            alpha_dict=alpha_dict_real # not used - passed "as dummy"
                        )
                else:
                    # This branch is evaluation on accepts, either biased or unbiased
                    # therefore they are also dependent on the metric used as basis
                    # for the decision and it must be considered in the thresholding
                    # calculation - which was done -. `th + "_" + m` makes sure we
                    # get the realized performance on the right accepts
                    realized = stats[perf+ '_' + th + "_" + m + '_real_' + m] # [gen, cv+1=M]
                    realized_in_cvs = (
                        # Step 1: Put gen at the end as expected by gen_lines_colors_and_main
                        realized.transpose(0,-1) # [cv+1, gen]
                        # Step 2: Do not consider the observation corresponding to the threshold on which
                        #         the final decision was made - this will be painted separately, the mean
                        #         of the CV performances might still be interesting
                        [:-1] # [cv, gen]
                        # Step 3: Generate an "artificial" dimension to avoid the performance based on the
                        # actual decision being labeled as the main, this way the dim=2 will paint only
                        # the mean of the performances across CVs
                        .unsqueeze(-2) # [cv, 1, gen]
                    )
                    realized_based_on_actual_accept_decision = realized[:, -1] # [gen]
                    if not only_related_exp or should_plot_perf:
                        lines_real, colors_real, _ = gen_lines_colors_and_main(
                            stat=realized_in_cvs.masked_fill(
                                mask_no_bads_among_acc[th+'_'+m][1:].view(1,1, -1), float('nan')
                            ),
                            sample_sizes=sample_sizes,
                            base_color=colors[perf],
                            main_lbl=labels_linetypes[perf],
                            alpha_dict=alpha_dict_real
                        )
                        _, _, main_real = gen_lines_colors_and_main(
                            stat=realized_based_on_actual_accept_decision.masked_fill(
                                mask_no_bads_among_acc[th+'_'+m][1:], float('nan')
                            ),
                            sample_sizes=sample_sizes,
                            base_color=colors[perf],
                            main_lbl=labels_linetypes[perf],
                            alpha_dict=alpha_dict_real
                        )
                    if should_plot_perf and add_real_bayes:
                        lines_real_bayes, colors_real_bayes, main_real_bayes = gen_lines_colors_and_main(
                        stat=bayes_stats[perf+ '_' + th + "_" + m + '_real_' + m][1:].masked_fill(
                                mask_no_bads_among_acc[th+'_'+m][1:], float('nan')
                            ),
                        sample_sizes=sample_sizes,
                        base_color=colors["real_bayes"],
                        main_lbl=labels_linetypes["real_bayes"],
                        alpha_dict=alpha_dict_real # not used - passed "as dummy"
                    )
                        
                if not only_related_exp or should_plot_perf:
                    line_collections_data[current_lbl]["lines"].extend(lines_real)
                    line_collections_data[current_lbl]["colors"].extend(colors_real)
                    line_collections_data[current_lbl]["mains"].append(main_real)
                if should_plot_perf and add_real_bayes:
                    line_collections_data[current_lbl]["lines"].extend(lines_real_bayes)
                    line_collections_data[current_lbl]["colors"].extend(colors_real_bayes)
                    line_collections_data[current_lbl]["mains"].append(main_real_bayes)
                
    return line_collections_data

def plot_lines_and_mains(
        line_collections_data: Dict[str, Dict[str, List]],
        sample_sizes: torch.Tensor,
        sup_title: str = "",
        figsize: Tuple[int, int] = (16,9),
        add_legend: bool = True,
        add_fig_legend: bool = True,
        only_related_exp: bool = True,
        diff_plot: bool = False
    ) -> None:
    min_s, max_s = sample_sizes[[0,-1]]
    margin_x = (max_s - min_s) * 0.01

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)

    dt_mapper = {
        "acc_based" : "Accepts based",
        "oracle_naive" : "Oracle naive",
        "oracle_comparable" : "Oracle comparable"
    }
    metric_mapper = {
        "ks" : "KS-statistic",
        "roc" : "AU-ROC"
    }


    for i_exp, exp in enumerate(EXPECTATION_TYPES):
        for i_m, m in enumerate(METRIC_CATEGORIES):
            current_lbl = exp + '_' + m
            ax = axes[i_m, i_exp]

            plot_dict = line_collections_data[current_lbl]

            ax.set_xlim(min_s - margin_x, max_s + margin_x)
            ax.set_ylim(0,0.85)

            ax.add_collection(LineCollection(plot_dict["lines"], colors=plot_dict["colors"], linewidths=1.0))

            for main_dict in plot_dict["mains"]:
                ax.plot(*main_dict["args"], **main_dict["kwargs"])

            if add_legend:
                ax.legend()

            ax.set_ylabel(metric_mapper[m])
            ax.set_xlabel("Count generated data")

            ax.set_title(dt_mapper[exp])

            ax.autoscale()

    line_types, labels_linetypes, color_map = linetypes_labels_and_colors_for_exp_real_plot(
        only_related_exp=only_related_exp,
        diff_plot=diff_plot
    )

    legend_elements = []
    if len(sup_title) > 0:
        fig.suptitle(sup_title, fontsize=16)

    colors_already_added = []

    for lt in line_types:
        c = color_map[lt]
        if c not in colors_already_added:
            legend_elements.append(Line2D([0], [0], color=c, lw=1.5, label=labels_linetypes[lt]))
            colors_already_added.append(c)

    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    #fig.tight_layout()

    plt.show()

### Exp-real diffs calculation

def moving_average_with_nans(x: torch.Tensor, W: int, dim: int = -1, with_tails: bool = True) -> torch.Tensor:
    assert x.dim() > 0

    x_ndim = x.dim()
    dim %= x_ndim

    # Move target dim to the end
    x = x.movedim(dim, -1)   # [..., N]

    if with_tails:
        pad_left = W//2
        pad_rigt = W - pad_left
        x = torch.nn.functional.pad(x, (pad_left, pad_rigt), mode='constant', value=float('nan'))

    N = x.size(-1)
    idx_for_ma_windows = torch.arange(N - W).unsqueeze(1) + torch.arange(W)

    # Extract windows
    windows = x[..., idx_for_ma_windows]   # shape [..., N-W+1, W]

    # Compute NaN‑aware mean
    ma = windows.nanmean(dim=-1)   # shape [..., N-W+1]


    return ma.movedim(-1, dim)

def calc_diffs(exp: torch.Tensor, real: torch.Tensor) -> torch.Tensor:

    diff_real = real - exp.mean(dim=(-1, -2))
    return diff_real


def generate_lines_for_diffs(
        sim_objs : Dict[str, torch.Tensor],
        add_real_bayes: bool = True,
        make_abs: bool = False,
        W: Optional[int] = None,
        use_bayes_with_missing: bool = True
    ):
    add_ma = W is not None
    if add_ma:
        W = int(W)

    
    only_related_exp: bool = True
    stats = sim_objs["stats"]
    
    bayes_stats_lbl = "perf_bayes_"
    if use_bayes_with_missing:
        bayes_stats_lbl += "missing_"
    bayes_stats_lbl += "stats"
    bayes_stats = sim_objs[bayes_stats_lbl]

    exp_to_real_diffs_complete = {
        "sample_sizes" : sim_objs["sample_sizes"],
        "diff_data" : {}
    }

    mask_no_bads_among_acc = {
        k : v == 0 for k, v in get_acc_defaults_counts_per_round(sim_objs).items()
    }
    exp_to_real_diffs = exp_to_real_diffs_complete["diff_data"]

    for exp in EXPECTATION_TYPES:
        for m in METRIC_CATEGORIES:
            current_lbl = exp + '_' + m
            exp_to_real_diffs[current_lbl] = {
                "logit" : {
                    "diffs" : None,
                    "ma_diffs" : None
                }
            }
            if add_real_bayes:
                exp_to_real_diffs[current_lbl]["bayes"] = {
                    "diffs" : None,
                    "ma_diffs" : None
                }
            
                
            # 1. CV x K lines - plot all cv lines of the expectation
            exp_lbl = exp+'_'+m+'_stats'

            th_is_oracle = exp.startswith("oracle")
            th = "oracle" if th_is_oracle else exp
            assert th in THRESHOLD_BASIS, "Threshold logic changed"

            for perf in REAL_PERFORMANCE_TYPES:
                perf_is_unb_future = perf=="unbiased_future"
                should_add = False
                if exp == 'acc_based':
                    should_add = perf == "biased_acc"
                elif exp == 'oracle_naive':
                    should_add = perf == "unbiased_future"
                elif exp=='oracle_comparable':
                    should_add = perf == 'unbiased_acc'

                
                if perf_is_unb_future:
                    real_lbl_stats = perf+ '_' + th
                    if not th_is_oracle:
                        # If not oracle, the available history also depend on the basis
                        # of the threshold metric and can/should generate different
                        # scores which of course **do** affect the realized performance
                        real_lbl_stats += "_" + m # Make sure based on same metric for acc-based
                    real_lbl_stats += '_real_' + m
                    realized_logit = stats[real_lbl_stats] # [gen]
                    realized_bayes = bayes_stats[real_lbl_stats][1:]
                else:
                    # This branch is evaluation on accepts, either biased or unbiased
                    # therefore they are also dependent on the metric used as basis
                    # for the decision and it must be considered in the thresholding
                    # calculation - which was done -. `th + "_" + m` makes sure we
                    # get the realized performance on the right accepts
                    real_lbl_stats = perf+ '_' + th + "_" + m + '_real_' + m
                    mask_make_nan = mask_no_bads_among_acc[th+'_'+m][1:]
                    realized_logit = stats[real_lbl_stats][:, -1].masked_fill(mask_make_nan, float('nan')) # [gen]
                    realized_bayes = bayes_stats[real_lbl_stats][1:].masked_fill(mask_make_nan, float('nan'))

                if should_add:
                    logit_diffs = calc_diffs(stats[exp_lbl], realized_logit)
                    if make_abs:
                        logit_diffs.abs_()


                    exp_to_real_diffs[current_lbl]["logit"]["diffs"] = logit_diffs
                    if add_ma:
                        exp_to_real_diffs[current_lbl]["logit"]["ma_diffs"] = moving_average_with_nans(logit_diffs, W, dim=-1, with_tails=True)

                    if add_real_bayes:
                        bayes_diffs = calc_diffs(bayes_stats[exp_lbl][:-1], realized_bayes)
                        if make_abs:
                            bayes_diffs.abs_()

                        exp_to_real_diffs[current_lbl]["bayes"]["diffs"] = bayes_diffs
                        if add_ma:
                            exp_to_real_diffs[current_lbl]["bayes"]["ma_diffs"] = moving_average_with_nans(bayes_diffs, W, dim=-1, with_tails=True)


    return exp_to_real_diffs_complete

def gen_lines_from_diff_dict(exp_to_real_diffs):
    _, _, colors = linetypes_labels_and_colors_for_exp_real_plot(
        only_related_exp=True
    )
    labels_linetypes = {"logit" : "Logit", "bayes" : "Perfect Bayes"}
    colors_for_diffs = {"logit" : colors["exp"], "bayes" : colors["exp_bayes"]}

    alpha_dict_real = {3: 0.1, 2: 0.3, 1 : 0.8}

    sample_sizes = exp_to_real_diffs["sample_sizes"]

    line_collections_data: Dict[str, Dict[str, List]] = {}

    for lbl, diff_data in exp_to_real_diffs["diff_data"].items():
        line_collections_data[lbl] = {
                "lines" : [],
                "colors" : [],
                "mains" : []
        }

        for model in ["logit", "bayes"]:
            if diff_data[model]["ma_diffs"] is None:
                _, _, main_real = gen_lines_colors_and_main(
                    stat=diff_data[model]["diffs"],
                    sample_sizes=sample_sizes,
                    base_color=colors_for_diffs[model],
                    main_lbl=labels_linetypes[model],
                    alpha_dict=alpha_dict_real
                )
                line_collections_data[lbl]["mains"].append(main_real)
            else:
                lines_diffs, colors_diffs, _ = gen_lines_colors_and_main(
                    stat=diff_data[model]["diffs"].unsqueeze(0),
                    sample_sizes=sample_sizes,
                    base_color=colors_for_diffs[model],
                    main_lbl=labels_linetypes[model],
                    alpha_dict=alpha_dict_real
                )
                _, _, main_ma = gen_lines_colors_and_main(
                    stat=diff_data[model]["ma_diffs"],
                    sample_sizes=sample_sizes,
                    base_color=colors_for_diffs[model],
                    main_lbl=labels_linetypes[model],
                    alpha_dict=alpha_dict_real
                )
                line_collections_data[lbl]["lines"].extend(lines_diffs)
                line_collections_data[lbl]["colors"].extend(colors_diffs)
                line_collections_data[lbl]["mains"].append(main_ma)



    return line_collections_data
    

## Diff areas calculation

def construct_all_diffs_tensor(
    all_sim_objs,
    biases: Dict[str, float],
    corrs: Dict[str, float],
    expectation_types: list,
    metric_categories: list,
    diff_types: list,
    classifiers: list
) -> Tuple[torch.Tensor, int, int, int, int, int]:
    """
    Construct the all_diffs tensor by iterating over parameter combinations.
    
    Args:
        grid_path: Path to simulation directory
        biases: Dictionary of bias values
        corrs: Dictionary of correlation values
        expectation_types: List of expectation types
        metric_categories: List of metric categories
        diff_types: List of diff types
        sample_sizes: Tensor of sample sizes
        
    Returns:
        Tuple of (all_diffs_tensor, B, Co, E, M, D) with shapes and counts
    """
    all_diffs = []
    
    for b_s in biases.keys():
        for c_s in corrs.keys():
            sim_objs = all_sim_objs[b_s][c_s]
            exp_to_real_diffs = generate_lines_for_diffs(sim_objs, make_abs=True, W=5)
            
            for e in expectation_types:
                for m in metric_categories:
                    case_lbl = e + '_' + m
                    case_data = exp_to_real_diffs["diff_data"][case_lbl]
                    
                    for classif in classifiers:
                        for dt in diff_types:
                            all_diffs.append(case_data[classif][dt])
    
    B = len(biases)
    Co = len(corrs)
    E = len(expectation_types)
    M = len(metric_categories)
    Cl = 2  # logit, bayes
    D = len(diff_types)
    sample_sizes = exp_to_real_diffs["sample_sizes"]
    N = sample_sizes.size(0)
    
    all_diffs = torch.stack(all_diffs, dim=0).reshape(B, Co, E, M, Cl, D, N)
    return all_diffs, sample_sizes, B, Co, E, M, D


def get_areas(
    all_diffs: torch.Tensor,
    sample_sizes: torch.Tensor
) -> torch.Tensor:
    """
    Compute areas under curves using trapezoidal integration.
    D is [abs_diffs, ma_diffs]
    
    Args:
        all_diffs: Tensor of shape [B, Co, E, M, Cl, D, N]
        sample_sizes: Tensor of sample sizes
        metric_categories: List of metric categories
        
    Returns:
        Areas tensor of shape [B, Co, E, M, Cl, D + 1]
    """
    mask_valid = ~all_diffs.isnan()
    areas_interpolated = masked_batched_trapz(
        y=all_diffs,
        x=sample_sizes.expand_as(all_diffs),
        mask=mask_valid,
        dim=-1
    )
    idx_abs_diffs = 0
    sum_performance = all_diffs[..., idx_abs_diffs, :].nansum(dim=-1, keepdim=True) # [B, Co, E, M, Cl, 1]
    return torch.cat(
            [
                areas_interpolated,
                sum_performance
            ],
            dim=-1
        ) # [B, Co, E, M, Cl, D + 1]


def normalize_areas(
    areas: torch.Tensor,
    valid_mask_diffs: torch.Tensor,
    sample_sizes: torch.Tensor,
    ordered_worse_per_cat: Dict[str, float] = {"ks": 1, "roc": 0.5}
) -> torch.Tensor:
    """
    Normalize areas by worst-case areas for each metric category.
    
    Args:
        areas: Tensor of shape [B, Co, E, M, Cl, D+1]
        valid_mask_diffs: Tensor of shape [B, Co, E, M, Cl, D, N]
        metric_categories: List of metric categories
        sample_sizes: Tensor of sample sizes
        
    Returns:
        Normalized areas tensor of shape [B, Co, E, M, Cl, D+1]
    """
    B, Co, E, M, Cl, D, N = valid_mask_diffs.shape
    if not (len(ordered_worse_per_cat) == M):
        raise RuntimeError("Wrong metric categories size")
    
    worst_case_ten = areas.new_tensor(list(ordered_worse_per_cat.values()))
    worst_case_ten_exp = worst_case_ten.unsqueeze(-1).expand(M, N)
    
    worst_possible_areas = masked_batched_trapz(
        y=worst_case_ten_exp,
        x=sample_sizes.expand(M, N),
        mask=torch.ones_like(worst_case_ten_exp, dtype=bool)
    ) # [M]

    mask_valid_abs_diffs = valid_mask_diffs[..., 0, :] # [B, Co, E, M, Cl, N]
    worst_possible_sums = torch.sum(
        worst_case_ten_exp.view(M, 1, N) * mask_valid_abs_diffs, # [B, Co, E, M, Cl, N]
        dim=-1
    )  # [B, Co, E, M, Cl]

    normed_areas = areas.clone() # [B, Co, E, M, Cl, D+1]
    normed_areas[..., :-1] /= worst_possible_areas.view(M, 1, 1) # div result shape = [B, Co, E, M, Cl, D]
    normed_areas[..., -1] /= worst_possible_sums
    
    return normed_areas


def plot_heatmap_grid(
    data: torch.Tensor,
    metric_categories: list,
    expectation_types: list,
    biases: Dict[str, float],
    corrs: Dict[str, float],
    cmap: str = 'RdBu_r',
    suptitle: Optional[str] = None,
    title_bold: bool = True,
    cbar_label: str = 'Logit - Bayes\n(Normalized Area)',
    title_mapping: Optional[Callable[[str, str], str]] = None
) -> None:
    M_len = len(metric_categories)
    E_len = len(expectation_types)

    # Sort biases and corrs for consistent ordering
    sorted_bias_keys = sorted(biases.keys(), key=lambda x: biases[x])
    sorted_corr_keys = sorted(corrs.keys(), key=lambda x: corrs[x])
    
    bias_key_list = list(biases.keys())
    corr_key_list = list(corrs.keys())
    sorted_bias_idxs = [bias_key_list.index(k) for k in sorted_bias_keys]
    sorted_corr_idxs = [corr_key_list.index(k) for k in sorted_corr_keys]

    # Labels
    bias_labels = [f'{biases[k]:.2f}' for k in sorted_bias_keys]
    corr_labels = [f'{corrs[k]:.2f}' for k in sorted_corr_keys]

    fig = plt.figure(figsize=(5 * E_len, 4 * M_len + 0.5))
    gs = GridSpec(
        M_len, E_len + 1,
        figure=fig,
        width_ratios=[1] * E_len + [0.05],
        wspace=0.3,
        hspace=0.4
    )
    
    im = None

    
    v_min = data.min().item()
    v_max = data.max().item()
    
    # Create heatmaps
    for m_idx, m_cat in enumerate(metric_categories):
        for e_idx, e_type in enumerate(expectation_types):
            ax = fig.add_subplot(gs[m_idx, e_idx])
            
            # Extract and reorder data
            raw_data = data[:, :, e_idx, m_idx].numpy()
            heatmap_data = raw_data[np.ix_(sorted_bias_idxs, sorted_corr_idxs)]
            
            # Plot heatmap
            im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=v_min, vmax=v_max)
            
            # Add black dots to negative values
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    if heatmap_data[i, j] < 0:
                        ax.plot(j, i, 'k.', markersize=8)
            
            # Set title
            if title_mapping is not None:
                title = title_mapping(e_type, m_cat)
            else:
                title = f'{e_type}_{m_cat}'
            
            title_weight = 'bold' if title_bold else 'normal'
            ax.set_title(title, fontsize=10, fontweight=title_weight)
            
            # Set labels
            ax.set_xlabel('Correlation', fontsize=9)
            ax.set_ylabel('Bias', fontsize=9)
            
            # Set tick labels
            ax.set_xticks(range(len(sorted_corr_keys)))
            ax.set_xticklabels(corr_labels, fontsize=8)
            
            ax.set_yticks(range(len(sorted_bias_keys)))
            ax.set_yticklabels(bias_labels, fontsize=8)
    
    # Add colorbar in its own subplot
    cbar_ax = fig.add_subplot(gs[:, E_len])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)
    
    # Add super title if provided
    if suptitle and len(suptitle) > 0:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    
    plt.show()

def plot_heatmap_grid_multiD(
    data: torch.Tensor,
    metric_categories: list,
    expectation_types: list,
    biases: Dict[str, float],
    corrs: Dict[str, float],
    cmap: str = "RdBu_r",
    suptitle: Optional[str] = None,
    title_bold: bool = True,
    cbar_label: str = "Logit - Bayes\n(Normalized Area)",
    title_mapping: Optional[Callable[[str, str], str]] = None,
    d_lbls: Optional[Iterable[str]] = None
) -> None:
    """
    data has shape

        [Bias, Corr, Expectation, Metric, D]

    Every heatmap cell is split horizontally into D equal pieces.
    """

    M_len = len(metric_categories)
    E_len = len(expectation_types)

    # ------------------------------------------------------------------
    # ordering
    # ------------------------------------------------------------------

    sorted_bias_keys = sorted(biases.keys(), key=lambda x: biases[x])
    sorted_corr_keys = sorted(corrs.keys(), key=lambda x: corrs[x])

    bias_key_list = list(biases.keys())
    corr_key_list = list(corrs.keys())

    sorted_bias_idxs = [bias_key_list.index(k) for k in sorted_bias_keys]
    sorted_corr_idxs = [corr_key_list.index(k) for k in sorted_corr_keys]

    bias_labels = [f"{biases[k]:.2f}" for k in sorted_bias_keys]
    corr_labels = [f"{corrs[k]:.2f}" for k in sorted_corr_keys]

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(5 * E_len, 4 * M_len + 0.5))
    D = data.shape[-1]

    d_lbls_is_not_none = d_lbls is not None
    if d_lbls_is_not_none:
        assert len(d_lbls) == D
    gs = GridSpec(
        M_len + int(d_lbls_is_not_none),
        E_len + 1,
        figure=fig,
        width_ratios=[1] * E_len + [0.05],
        height_ratios= [1] * M_len + [0.05] * int(d_lbls_is_not_none),
        wspace=0.3,
        hspace=0.4,
    )

    data_np = data.numpy() # [B, Co, E, M, D]

    vmin = data_np.min()
    vmax = data_np.max()

    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    

    # ------------------------------------------------------------------
    # draw all heatmaps
    # ------------------------------------------------------------------

    for m_idx, m_cat in enumerate(metric_categories):

        for e_idx, e_type in enumerate(expectation_types):

            ax = fig.add_subplot(gs[m_idx, e_idx])

            heatmap_data = data_np[:, :, e_idx, m_idx, :] # [B, Co, D]
            heatmap_data = heatmap_data[
                np.ix_(sorted_bias_idxs, sorted_corr_idxs, np.arange(D))
            ]

            n_bias, n_corr, _ = heatmap_data.shape

            width = 1.0 / D

            for i in range(n_bias):
                for j in range(n_corr):

                    for d in range(D):

                        value = heatmap_data[i, j, d]

                        rect = Rectangle(
                            (j - 0.5 + d * width, i - 0.5),
                            width,
                            1,
                            facecolor=cmap_obj(norm(value)),
                            edgecolor="none",
                        )

                        ax.add_patch(rect)

                        if value < 0:
                            ax.plot(
                                j - 0.5 + (d + 0.5) * width,
                                i,
                                "k.",
                                markersize=5,
                            )

            # draw grid
            for x in range(n_corr + 1):
                ax.axvline(x - 0.5, color="black", lw=0.8)

            for y in range(n_bias + 1):
                ax.axhline(y - 0.5, color="black", lw=0.8)

            # optional separators between D pieces
            if D > 1:
                for x in range(n_corr):
                    for d in range(1, D):
                        ax.axvline(x=x - 0.5 + d * width, ymin = -0.5, ymax=n_bias - 0.5,
                                   color='k', alpha=.5, lw=0.5, ls="--")

            ax.set_xlim(-0.5, n_corr - 0.5)
            ax.set_ylim(n_bias - 0.5, -0.5)

            ax.set_xticks(range(n_corr))
            ax.set_xticklabels(corr_labels, fontsize=8)

            ax.set_yticks(range(n_bias))
            ax.set_yticklabels(bias_labels, fontsize=8)

            ax.set_xlabel("Correlation", fontsize=9)
            ax.set_ylabel("Bias", fontsize=9)

            if title_mapping is None:
                title = f"{e_type}_{m_cat}"
            else:
                title = title_mapping(e_type, m_cat)

            ax.set_title(
                title,
                fontsize=10,
                fontweight="bold" if title_bold else "normal",
            )

    # ------------------------------------------------------------------
    # colorbar
    # ------------------------------------------------------------------

    cbar_ax = fig.add_subplot(gs[:, E_len])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label)

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

    if d_lbls_is_not_none:
        ax = fig.add_subplot(gs[-1, :])
        diffs_as_str = "".join([l + (', ' if i < D-2 else " and " if i == D-2 else "") for i, l in enumerate(d_lbls)])
        ax.axis('off')
        ax.text(
            y=0, x=0,
            s = (
                "Each subplot shows the data acoording to differences of " +
                diffs_as_str
            )
        )


    plt.show()

def wrapper_plot_grid_of_diffs_between_logit_and_acc(all_sim_objs, grid_biases, grid_corrs):
    diff_types = ["diffs", "ma_diffs"]
    # Construct all_diffs tensor
    ## [B, Co, E, M, Cl, D, N], [N]
    all_diffs_new, sample_sizes, B, Co, E, M, D = construct_all_diffs_tensor(
        all_sim_objs,
        biases=grid_biases,
        corrs=grid_corrs,
        expectation_types=EXPECTATION_TYPES,
        metric_categories=METRIC_CATEGORIES,
        diff_types=diff_types,
        classifiers=["logit", "bayes"]
    )

    # Get areas
    areas = get_areas(all_diffs_new, sample_sizes)

    # Normalize areas
    normed_areas = normalize_areas(areas, ~all_diffs_new.isnan(), sample_sizes)

    # # Compute logit - bayes difference
    logit_area_minus_bayes_normed = normed_areas[..., 0, :] - normed_areas[..., 1, :]


    def custom_title_mapping(e_type: str, m_cat: str) -> str:
        e_type_map = {
            "acc_based" : "Accepts based",
            "oracle_naive" : "Oracle naive",
            "oracle_comparable" : "Oracle comparable"
        }
        m_cat_mapper = {
            "roc" : "AUROC",
            "ks" : "KS"
        }
        return f'{e_type_map[e_type]} ({m_cat_mapper[m_cat]})'

    plot_heatmap_grid_multiD(
        data=logit_area_minus_bayes_normed,
        metric_categories=METRIC_CATEGORIES,
        expectation_types=EXPECTATION_TYPES,
        biases=grid_biases,
        corrs=grid_corrs,
        cmap="RdBu_r",
        suptitle="",
        title_bold=True,
        cbar_label="Logit - Bayes\n(Normalized Area)",
        title_mapping=custom_title_mapping,
        d_lbls=["Absolute", "Moving Average", "Pointwise sum"]
    )


## Accept rates

def plot_acc_rates_per_round(sim_objs, fig_title: str = ""):
    counts_per_round_base, th_meaning = simulation_counts_per_round(sim_objs) # [TT, AR, TBG, G]

    total_acc_count_per_threshold_and_round = counts_per_round_base[1:, 0, 0, 1:]
    total_applicants_per_round = counts_per_round_base[0, 0, 0, 1:]

    acc_rate_per_threshold_and_round = total_acc_count_per_threshold_and_round/total_applicants_per_round

    thresholds_orig_order = th_meaning[1:]
    sorted_indices_thresholds = sorted(range(len(thresholds_orig_order)), key=thresholds_orig_order.__getitem__)
    sorted_thresholds = [th_meaning[i+1] for i in sorted_indices_thresholds]
    sorted_acc_rates = acc_rate_per_threshold_and_round[sorted_indices_thresholds]

    count_rounds = acc_rate_per_threshold_and_round.size(-1)

    arange_R_np = torch.arange(1, count_rounds+1).numpy()

    fig_w, fig_h = 12, 9
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        wspace=0.2,
        hspace=0.3,
        bottom=0.1
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    th_mapper = {
        "acc_based_ks" : "Accepts Based (KS)",
        "acc_based_roc" : "Accepts Based (ROC-Optimal)",
        "oracle_ks" : "Oracle Comparable (KS)",
        "oracle_roc" : "Oracle Comparable (ROC-Optimal)"
    }

    for th, acceptance_rate, ax in zip(sorted_thresholds, sorted_acc_rates, axes):
        ax.plot(arange_R_np, acceptance_rate.numpy(), label="Acceptance Rate (AR)")

        window = 5
        ax.plot(arange_R_np, moving_average_with_nans(acceptance_rate, W=window, with_tails=True).numpy(),
                label="MA(AR), W="+str(window), ls="--")
        
        ax.axhline(y=sim_objs["data_generator"].prob_bad,
                color='black', linestyle='--', linewidth=1.5, label="$\\mathbb{P}(Y=b)$")


        ax.set_title(th_mapper[th])

        ax.set_xlabel("Acceptance Round")
        ax.set_ylabel("Acceptance Rate")

        #ax.legend()

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc = "lower center",
        ncol=len(handles)
    )

    if len(fig_title) > 0:
        fig.suptitle(fig_title, fontsize=16)

    plt.show()