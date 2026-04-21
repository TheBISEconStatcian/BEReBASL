import os
import sys
base_path = os.path.abspath("./..")
if base_path not in sys.path:
    sys.path.append(base_path)

from math import comb
from itertools import combinations

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import torch

from berebasl.simulation.credit_data_simulation import CreditDataGenerator

from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple

def make_plot_dicts_to_show_dgp(
        data_gen: CreditDataGenerator, 
        sample_size: int, 
        cmap_dict: Optional[Dict[str, Callable[[int], Tuple[float, float, float]]]]
    ) -> Union[List[Dict[str, List[Dict[str, Any]]]]]:
    sampled_feats, sampled_labels, sampled_K_idxs = data_gen.sample(sample_size, reveal_mixture_components=True)

    gb_vals = ["bad", "good"]
    plot_dicts = [{gb: [] for gb in gb_vals} for _ in range(data_gen.B)]

    is_batched = data_gen.is_batched
    is_mixture: Dict[str, bool] = {gb : getattr(data_gen, gb+'_mixture').is_mixture for gb in gb_vals}
    is_mixture["any"] = any(is_mixture.values())
    noise_var = data_gen.noise_std ** 2

    mixtures_K: Dict[str, Optional[int]] = {gb: getattr(data_gen, gb+'_mixture').K if is_mixture[gb] else None for gb in gb_vals}

    gm_params = sum([
        [gm.mean, gm.effective_cov_additive(noise_var) if data_gen.add_noise else gm.cov] 
        for gm in (data_gen.bad_mixture, data_gen.good_mixture)
    ], [])

    batch_iter = (
        (feats, lbls, K_idxs, gm_bad_mean, gm_bad_cov, gm_good_mean, gm_good_cov)
        for feats, lbls, K_idxs, gm_bad_mean, gm_bad_cov, gm_good_mean, gm_good_cov
        in (
            zip(
                sampled_feats, 
                sampled_labels, 
                sampled_K_idxs if is_mixture["any"] else [None]*data_gen.B, 
                *gm_params
            )
            if is_batched else
            [(
                sampled_feats, 
                sampled_labels, 
                sampled_K_idxs, 
                *gm_params
            )]
        )
    )
    F = data_gen.features_count
    for b, (B_feats, B_lbls, B_K_idxs, B_gm_bad_mean, B_gm_bad_cov, B_gm_good_mean, B_gm_good_cov) in enumerate(batch_iter):
        plot_dict = plot_dicts[b]
        batch_add_suffix = f"^{{b={b}}}" if is_batched else ""
        for f1, f2 in combinations(range(F), 2):
            for gb in gb_vals:
                gb_add_suffix = gb[0] + batch_add_suffix
                for k in range(mixtures_K[gb]) if is_mixture[gb] else [0]:
                    mask = B_lbls == data_gen.bad_good_encoding[gb]
                    mean, cov = (B_gm_bad_mean, B_gm_bad_cov) if gb=="bad" else (B_gm_good_mean, B_gm_good_cov)
                    # Extract current variable components
                    mean = mean[..., [f1, f2]]
                    cov = cov[..., [f1, f2], :][..., [f1, f2]]
                    if is_mixture[gb]:
                        mask &= B_K_idxs == k
                        mean, cov = [p[k] for p in (mean, cov)]
                        suffix = gb_add_suffix + "_{k=" + str(k) + '}'
                    else:
                        suffix = gb_add_suffix

                    plot_dict[gb].append({
                        "sample": B_feats[:, [f1, f2]][mask],
                        "mixture_mean": mean,
                        "mixture_cov": cov,
                        "color": cmap_dict[gb](k),
                        "label_suffix": suffix,
                        "var_idxs" : (f1, f2)
                    })
    if not is_batched:
        plot_dicts = plot_dicts[0]

    return plot_dicts

def plot_cov_ellipse(mean, cov, ax, color_alpha: float = 0.2):
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues (largest first for consistency)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Compute angle (in degrees) of the ellipse
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Width and height = 2 * sqrt(eigenvalues)
    # (factor 2 because matplotlib expects full width, not radius)
    width, height = 2 * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=(0,0,0,color_alpha),  # Semi-transparent black
        facecolor='none',
        linewidth=1.5,
        zorder = 10.0
    )

    ax.add_patch(ellipse)

def plot_population_sample(
        sample: torch.Tensor,
        mixture_mean: torch.Tensor,
        mixture_cov: torch.Tensor,
        color: Tuple[int, int, int],
        label_suffix: str,
        ax: plt.Axes = None,
        transparency_alpha: float = 0.08
) -> None:
    if ax is None:
        ax = plt.gca()

    ax.scatter(sample[:,0], sample[:,1], s=5, color=color + (transparency_alpha,), marker='o')
    ax.scatter(mixture_mean[0].item(), mixture_mean[1].item(), s=20, color = color,
                facecolor='white', edgecolor=color,
                label=f"$\\mathbb{{E}}\\,[X_{{{label_suffix}}}]$", zorder = 10.0)
    plot_cov_ellipse(
        mixture_mean.cpu().numpy(), 
        mixture_cov.cpu().numpy(),
        ax
    )

def place_legend_below_axis(
    ax: plt.Axes,
    ncol: Optional[int] = None,
    y_offset: float = -0.18,
    fontsize: int = 9
):
    """
    Places a wide, row-wise legend below the x-axis.

    Parameters
    ----------
    ax : plt.Axes
    ncol : number of columns (defaults to all entries in one row)
    y_offset : vertical offset (negative -> below axis)
    fontsize : legend font size
    """
    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    # Default: make it as wide as possible (single row)
    if ncol is None:
        ncol = len(handles)

    ax.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        columnspacing=1.2
    )

def plot_dgp(
        data_gen: CreditDataGenerator, 
        sample_size: int, 
        cmap_dict: Optional[Dict[str, Callable[[int], Tuple[float, float, float]]]] = None,
        axes: Optional[List[Dict[str, plt.Axes]]] = None,
        plotting_order = ["good", "bad"],
        transparency_alpha = 0.08,
        legend_y_offset = -0.09,
        legend_modus: Literal['none', 'below', 'default'] = "below"
    ) -> None:
    nrows_per_F = {
        2: 1,
        3: 1,
        4: 2,
        5: 5,
        6: 5,
        7: 7,
        8: 7
    }
    if cmap_dict is None:
        if max(data_gen.good_mixture.K, data_gen.bad_mixture.K) > 5:
            raise AssertionError("Too many mixtures - plotting only possible by passing cmap_dict, also consider using other method")
        tab10_good_bad_idxs = {
            "bad" : [1, 3, 4, 5, 7],
            "good" : [0, 2, 6, 8, 9]
        }

        cmap_dict: Dict[str, Callable[[int], Tuple[float, float, float]]] = {
            "bad" : lambda idx : colormaps["tab10"](tab10_good_bad_idxs["bad"][idx])[:3],
            "good" : lambda idx : colormaps["tab10"](tab10_good_bad_idxs["good"][idx])[:3]
        }
    
    plot_dicts = make_plot_dicts_to_show_dgp(data_gen, sample_size, cmap_dict)
    is_batched = data_gen.is_batched
    B = data_gen.B
    F = data_gen.features_count

    if not is_batched:
        plot_dicts = [plot_dicts]

    for b, plot_dict_list, aes_dict in zip(range(B), plot_dicts, [None]*len(plot_dicts) if axes is None else axes):
        if aes_dict is None:
            ncols = comb(F, 2) / nrows_per_F[F]
            nrows = nrows_per_F[F]
            fig, aes = plt.subplots(nrows=nrows, ncols=int(ncols),
                                    figsize=(5*ncols, 3*nrows))
            aes_dict = {}
            for (f1, f2), a in zip(combinations(range(F), 2), aes.flatten() if F>2 else [aes]):
                a.set_xlabel(f"$x_{{{f1}}}$")
                a.set_ylabel(f"$x_{{{f2}}}$")
                aes_dict[f"{f1:02}{f2:02}"] = a

            fig_title = r"Visualization of $\left\{ X_i"
            if is_batched:
                fig_title += f"^{{B={b}}}"
            
            fig_title += r" \right\}_{i=0}^{" + str(F-1) + "}$"
            fig.suptitle(fig_title)

        # Plot all data into the appropriate axes
        for current_key in plotting_order:
            for plot_dict in plot_dict_list[current_key]:
                f1, f2 = plot_dict.pop("var_idxs")
                #print(plot_dict["sample"].mean(dim=0), plot_dict["mixture_mean"])
                plot_population_sample(ax=aes_dict[f"{f1:02}{f2:02}"], transparency_alpha=transparency_alpha, **plot_dict)

        # Collect handles and labels from all axes for shared legend
        all_handles, all_labels = [], []
        for ax in aes_dict.values():
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in all_labels:  # Avoid duplicates
                    all_handles.append(h)
                    all_labels.append(l)

        # Place shared legend below the figure
        if legend_modus == "below" and all_handles:
            ncol = max(max(data_gen.good_mixture.K, round((data_gen.good_mixture.K + data_gen.bad_mixture.K)/2)), 4) if data_gen.good_mixture.is_mixture else None
            if ncol is None:
                ncol = len(all_handles)
            
            fig.legend(
                all_handles,
                all_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, legend_y_offset - 0.05),
                ncol=ncol,
                frameon=False,
                fontsize=9,
                handlelength=1.5,
                columnspacing=1.2
            )

        if axes is None:
            plt.show()
