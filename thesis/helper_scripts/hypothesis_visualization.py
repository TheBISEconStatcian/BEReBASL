from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from typing import Any, Dict, List, Tuple, Union
from experiments.acceptance_loop_cv_based_MNAR import EXPECTATION_TYPES, METRIC_CATEGORIES, THRESHOLD_BASIS, REAL_PERFORMANCE_TYPES
from matplotlib import pyplot as plt

import torch

from numpy import ndarray

from berebasl.simulation.credit_data_simulation import CreditData

def gen_lines_colors_and_main(
        stat: torch.Tensor, 
        sample_sizes: torch.Tensor, 
        base_color: Tuple[float],
        main_lbl: str, 
        alpha_dict: Dict[int, float], 
        lines: List[ndarray] = None, 
        colors: List[Tuple[float]] = None,
        sort_stat: bool = True
    )-> Tuple[List[ndarray], List[Tuple[float]], Dict[str, Union[List[ndarray], Dict[str, Any]]]]:
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

def get_acc_defaults_counts_per_round(sim_objs) -> Dict[str, torch.Tensor]:
    all_threshold_lbls = [t+'_'+m for m in METRIC_CATEGORIES for t in THRESHOLD_BASIS]
    acc_flags: Dict[str, torch.Tensor] = {k : v.clone() for k, v in sim_objs['alternative_accepted'].items()}
    lbl_acc_flag_in_data = set(all_threshold_lbls) - set(acc_flags.keys())
    assert len(lbl_acc_flag_in_data) == 1
    lbl_acc_flag_in_data = next(lbl_acc_flag_in_data.__iter__())

    credit_data: CreditData = sim_objs["credit_data"]
    acc_flags[lbl_acc_flag_in_data] = credit_data.accepted.clone()

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

def generate_line_collections_for_exp_real_plot(
        sim_objs : Dict[str, torch.Tensor],
        only_related_exp: bool = True,
        add_real_bayes: bool = True
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
    bayes_stats = sim_objs["perf_bayes_stats"]
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

from typing import Optional

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
        W: Optional[int] = None
    ):
    add_ma = W is not None
    if add_ma:
        W = int(W)

    
    only_related_exp: bool = True
    stats = sim_objs["stats"]
    bayes_stats = sim_objs["perf_bayes_stats"]

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
    
