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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import torch

from berebasl.simulation.credit_data_simulation import CreditDataGenerator, GaussianMixture

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union, Tuple

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
        batch_add_suffix = f"^{{B={b}}}" if is_batched else ""
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


def make_plot_dicts_to_show_dgp_3d(
        data_gen: CreditDataGenerator,
        sample_size: int,
        cmap_dict: Optional[Dict[str, Callable[[int], Tuple[float, float, float]]]]
    ) -> Union[List[Dict[str, List[Dict[str, Any]]]]]:
    sampled_feats, sampled_labels, sampled_K_idxs = data_gen.sample(sample_size, reveal_mixture_components=True)

    gb_vals = ["bad", "good"]
    plot_dicts = [{gb: [] for gb in gb_vals} for _ in range(data_gen.B)]

    is_batched = data_gen.is_batched
    is_mixture: Dict[str, bool] = {gb: getattr(data_gen, gb + '_mixture').is_mixture for gb in gb_vals}
    is_mixture["any"] = any(is_mixture.values())
    noise_var = data_gen.noise_std ** 2

    mixtures_K: Dict[str, Optional[int]] = {gb: getattr(data_gen, gb + '_mixture').K if is_mixture[gb] else None for gb in gb_vals}

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
                sampled_K_idxs if is_mixture["any"] else [None] * data_gen.B,
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

    for b, (B_feats, B_lbls, B_K_idxs, B_gm_bad_mean, B_gm_bad_cov, B_gm_good_mean, B_gm_good_cov) in enumerate(batch_iter):
        plot_dict = plot_dicts[b]
        batch_add_suffix = f"^{{b={b}}}" if is_batched else ""
        for gb in gb_vals:
            gb_add_suffix = gb[0] + batch_add_suffix
            for k in range(mixtures_K[gb]) if is_mixture[gb] else [0]:
                mask = B_lbls == data_gen.bad_good_encoding[gb]
                mean, cov = (B_gm_bad_mean, B_gm_bad_cov) if gb == "bad" else (B_gm_good_mean, B_gm_good_cov)
                if is_mixture[gb]:
                    mask &= B_K_idxs == k
                    mean, cov = [p[k] for p in (mean, cov)]
                    suffix = gb_add_suffix + "_{k=" + str(k) + '}'
                else:
                    suffix = gb_add_suffix

                plot_dict[gb].append({
                    "sample": B_feats[mask],
                    "mixture_mean": mean,
                    "mixture_cov": cov,
                    "color": cmap_dict[gb](k),
                    "label_suffix": suffix
                })

    if not is_batched:
        plot_dicts = plot_dicts[0]

    return plot_dicts


def plot_cov_ellipses(mean, cov, ax, probs: Iterable[float] = (0.5, 0.8, 0.95),
                      edgecolor="k", facecolor="none",
                      linewidth=0.8, alpha=0.6):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    legend_handle = None
    coverage_label = (
        r"Coverage prob."
        + r"$\in \{" + ", ".join([(f"{pr:.2f}")[1:] for pr in probs]) + r"\}$"
    )
    for p in probs:
        r = np.sqrt(chi2.ppf(p, df=2))
        width, height = 2 * r * np.sqrt(eigvals)

        ell = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=10,
            label=None
        )
        ax.add_patch(ell)

        if legend_handle is None:
            legend_handle = Line2D(
                [], [],
                color=edgecolor,
                marker='o',
                linestyle='None',
                linewidth=linewidth,
                alpha=alpha,
                markerfacecolor='none',
                markeredgewidth=linewidth,
                markersize=8,
                label=coverage_label
            )

    return legend_handle

def plot_population_sample(
        sample: torch.Tensor,
        mixture_mean: torch.Tensor,
        mixture_cov: torch.Tensor,
        color: Tuple[int, int, int],
        label_suffix: str,
        ax: plt.Axes = None,
        transparency_alpha: float = 0.08,
        ellipses_probs: Iterable[float] = (0.5,0.8,.95),
        ellipses_width: float = .8,
        ellipses_alpha: float = .6
) -> None:
    if ax is None:
        ax = plt.gca()

    ax.scatter(sample[:,0], sample[:,1], s=5, color=color + (transparency_alpha,), marker='o')
    ax.scatter(mixture_mean[0].item(), mixture_mean[1].item(), s=20, color = color,
                facecolor='white', edgecolor=color,
                label=f"$\\mathbb{{E}}\\,[X_{{{label_suffix}}}]$", zorder = 10.0)
    coverage_handle = plot_cov_ellipses(
        mixture_mean.cpu().numpy(), 
        mixture_cov.cpu().numpy(),
        ax,
        probs=ellipses_probs,
        linewidth=ellipses_width,
        alpha=ellipses_alpha
    )
    if coverage_handle is not None and not hasattr(ax, '_coverage_legend_handle'):
        ax._coverage_legend_handle = coverage_handle


def plot_cov_ellipsoid_3d(mean, cov, ax, probs: Iterable[float] = (0.8,),
                       edgecolor="k", linewidth=0.8, alpha=0.18,
                       n_points: int = 24):
    """Plot a 3D ellipsoid at a given covariance level and return a legend handle."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    legend_handle = None
    coverage_label = (
        r"Coverage prob. "
        + r"$p \in \{" + ", ".join([(f"{pr:.2f}")[1:] for pr in probs]) + r"\}$"
    )

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)

    for p in probs:
        r = np.sqrt(chi2.ppf(p, df=3))
        radii = r * np.sqrt(eigvals)
        ellipsoid = eigvecs @ np.diag(radii) @ sphere
        xs = ellipsoid[0].reshape(x.shape) + mean[0]
        ys = ellipsoid[1].reshape(x.shape) + mean[1]
        zs = ellipsoid[2].reshape(x.shape) + mean[2]

        ax.plot_wireframe(
            xs, ys, zs,
            color=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            rstride=max(1, n_points // 12),
            cstride=max(1, n_points // 12)
        )

        if legend_handle is None:
            legend_handle = Line2D(
                [], [],
                color=edgecolor,
                marker='o',
                linestyle='None',
                linewidth=linewidth,
                alpha=alpha,
                markerfacecolor='none',
                markeredgewidth=linewidth,
                markersize=8,
                label=coverage_label
            )

    return legend_handle


def plot_population_sample_3d(
        sample: torch.Tensor,
        mixture_mean: torch.Tensor,
        mixture_cov: torch.Tensor,
        color: Tuple[int, int, int],
        label_suffix: str,
        ax: plt.Axes = None,
        transparency_alpha: float = 0.08,
        ellipsoid_probs: Iterable[float] = (0.8,),
        ellipsoid_width: float = .8,
        ellipsoid_alpha: float = .18
) -> None:
    if ax is None:
        ax = plt.gca()

    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=5,
               color=color + (transparency_alpha,), marker='o')
    ax.scatter(mixture_mean[0].item(), mixture_mean[1].item(), mixture_mean[2].item(),
               s=30, color=color, facecolor='white', edgecolor=color,
               label=f"$\\mathbb{{E}}\\,[X_{{{label_suffix}}}]$", zorder=10.0)

    coverage_handle = plot_cov_ellipsoid_3d(
        mixture_mean.cpu().numpy(),
        mixture_cov.cpu().numpy(),
        ax,
        probs=ellipsoid_probs,
        edgecolor=color,
        linewidth=ellipsoid_width,
        alpha=ellipsoid_alpha,
        n_points=18
    )
    if coverage_handle is not None and not hasattr(ax, '_coverage_legend_handle'):
        ax._coverage_legend_handle = coverage_handle


def plot_credit_dgp_3d(
        data_gen: CreditDataGenerator,
        sample_size: int,
        cmap_dict: Optional[Dict[str, Callable[[int], Tuple[float, float, float]]]] = None,
        axes: Optional[List[plt.Axes]] = None,
        plotting_order = ["good", "bad"],
        fig_title: Optional[str] = None,
        transparency_alpha = 0.08,
        legend_y_offset = -0.09,
        width: float = 8.0,
        height: float = 6.0,
        return_figures: bool = False,
        ellipsoid_probs: Iterable[float] = (0.8,),
        ellipsoid_width: float = .8,
        ellipsoid_alpha: float = .18,
        elev: int = 20,
        azim: int = 35
    ) -> Optional[List[plt.Figure]]:
    """Visualize a 3D MVN or Gaussian mixture for F=3 using the same colour and legend semantics as 2D."""
    if data_gen.features_count != 3:
        raise ValueError("plot_credit_dgp_3d only supports F=3")
    
    if cmap_dict is None:
        cmap_dict = _default_cmap_dict(
            K_good=data_gen.good_mixture.K,
            K_bad=data_gen.bad_mixture.K
        )
    plot_dicts = make_plot_dicts_to_show_dgp_3d(data_gen, sample_size, cmap_dict)

    is_batched = data_gen.is_batched
    B = data_gen.B
    if not is_batched:
        plot_dicts = [plot_dicts]

    created_figures = [] if (axes is None and return_figures) else None
    axes_list = [None] * len(plot_dicts) if axes is None else axes

    for b, plot_dict_list, ax in zip(range(B), plot_dicts, axes_list):
        if ax is None:
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111, projection='3d')
            if created_figures is not None:
                created_figures.append(fig)
            if fig_title is None:
                fig_title = r"3D visualization of $\{X_i"
                if is_batched:
                    fig_title += f"^{{B={b}}}"
                fig_title += r"\}_{i=0}^{2}$"
            fig.suptitle(fig_title)
        else:
            fig = ax.figure

        for current_key in plotting_order:
            for plot_dict in plot_dict_list[current_key]:
                plot_population_sample_3d(
                    sample=plot_dict["sample"],
                    mixture_mean=plot_dict["mixture_mean"],
                    mixture_cov=plot_dict["mixture_cov"],
                    color=plot_dict["color"],
                    label_suffix=plot_dict["label_suffix"],
                    ax=ax,
                    transparency_alpha=transparency_alpha,
                    ellipsoid_probs=ellipsoid_probs,
                    ellipsoid_width=ellipsoid_width,
                    ellipsoid_alpha=ellipsoid_alpha
                )

        ax.set_xlabel(r"$x_{0}$")
        ax.set_ylabel(r"$x_{1}$")
        ax.set_zlabel(r"$x_{2}$")
        ax.view_init(elev=elev, azim=azim)

        all_handles, all_labels = [], []
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

        coverage_handle = getattr(ax, '_coverage_legend_handle', None)
        if coverage_handle is not None and coverage_handle.get_label() not in all_labels:
            all_handles.append(coverage_handle)
            all_labels.append(coverage_handle.get_label())

        if all_handles:
            fig.legend(
                all_handles,
                all_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, legend_y_offset),
                ncol=max(1, min(len(all_handles), 4)),
                frameon=False,
                fontsize=9,
                handlelength=1.5,
                columnspacing=1.2
            )

        if axes is None and not return_figures:
            plt.show()

    return created_figures


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

def _default_cmap_dict(K_good: int, K_bad: int) -> Dict[str, Callable[[int], Tuple[float, float, float]]]:
    if max(K_good, K_bad) > 5:
        raise AssertionError("Too many mixtures - plotting only possible by passing cmap_dict, also consider using other method")
    tab10_good_bad_idxs = {
        "bad" : [1, 3, 4, 5, 7],
        "good" : [0, 2, 6, 8, 9]
    }

    cmap_dict: Dict[str, Callable[[int], Tuple[float, float, float]]] = {
        "bad" : lambda idx : colormaps["tab10"](tab10_good_bad_idxs["bad"][idx])[:3],
        "good" : lambda idx : colormaps["tab10"](tab10_good_bad_idxs["good"][idx])[:3]
    }

    return cmap_dict

def plot_credit_dgp_pairwise(
        data_gen: CreditDataGenerator, 
        sample_size: int, 
        cmap_dict: Optional[Dict[str, Callable[[int], Tuple[float, float, float]]]] = None,
        axes: Optional[List[Dict[str, plt.Axes]]] = None,
        plotting_order = ["good", "bad"],
        fig_title: Optional[str] = None,
        transparency_alpha = 0.08,
        legend_y_offset = -0.09,
        width_per_axcol: float = 5.0,
        height_per_axrow: float = 3.0,
        vspace_title_and_legend: float = 0.2,
        return_figures: bool = False,
        ellipses_probs: Iterable[float] = (.5,.8,.95),
        ellipses_width: float = .8,
        ellipses_alpha: float = .6
    ) -> Optional[List[plt.Figure]]:
    """
    Visualize all pairwise feature relationships for bad/good classes.
    
    Plots samples and Gaussian mixture components for each pair of features.
    When ``axes`` is ``None``, figures are auto-created; pass ``return_figures=True``
    to get them for composition instead of calling ``plt.show()``.
    
    Args:
        data_gen: CreditDataGenerator instance to visualize
        sample_size: Number of samples to draw for visualization
        cmap_dict: Optional color mapping for classes; auto-generated if None
        axes: List of axis dicts (one per batch); if None, creates figures
        plotting_order: Order to plot classes (default: ["good", "bad"])
        fig_title: Optional custom figure title; auto-generated if None
        transparency_alpha: Scatter plot transparency (default: 0.08)
        legend_y_offset: Vertical offset for legend below figure (default: -0.09)
        width_per_axcol: Width per axis column in inches (default: 5.0)
        height_per_axrow: Height per axis row in inches (default: 3.0)
        vspace_title_and_legend: Vertical spacing for title/legend (default: 0.2)
        return_figures: If ``True`` and ``axes is None``, return figures instead of showing
    
    Returns:
        ``List[plt.Figure]`` if ``return_figures=True`` and ``axes is None``, else ``None``
    """
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
        cmap_dict = _default_cmap_dict(
            K_good=data_gen.good_mixture.K,
            K_bad=data_gen.bad_mixture.K
        )
    
    plot_dicts = make_plot_dicts_to_show_dgp(data_gen, sample_size, cmap_dict)
    is_batched = data_gen.is_batched
    B = data_gen.B
    F = data_gen.features_count

    if not is_batched:
        plot_dicts = [plot_dicts]

    created_figures = [] if (axes is None and return_figures) else None
    
    for b, plot_dict_list, aes_dict in zip(range(B), plot_dicts, [None]*len(plot_dicts) if axes is None else axes):
        if aes_dict is None:
            ncols = comb(F, 2) / nrows_per_F[F]
            nrows = nrows_per_F[F]
            fig, aes = plt.subplots(
                nrows=nrows, ncols=int(ncols),
                figsize=(
                    width_per_axcol*ncols, 
                    height_per_axrow*nrows + vspace_title_and_legend
                )
            )
            if created_figures is not None:
                created_figures.append(fig)
            
            aes_dict = {}
            for (f1, f2), a in zip(combinations(range(F), 2), aes.flatten() if F>2 else [aes]):
                a.set_xlabel(f"$x_{{{f1}}}$")
                a.set_ylabel(f"$x_{{{f2}}}$")
                aes_dict[f"{f1:02}{f2:02}"] = a

            if fig_title is None:
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
                plot_population_sample(
                    ax=aes_dict[f"{f1:02}{f2:02}"],
                    transparency_alpha=transparency_alpha,
                    ellipses_probs=ellipses_probs,
                    ellipses_width=ellipses_width,
                    ellipses_alpha=ellipses_alpha,
                    **plot_dict
                )

        # Collect handles and labels from all axes for shared legend
        # and Add coverage legend handle last if present
        coverage_handle = None
        coverage_label = None
        all_handles, all_labels = [], []
        for ax in aes_dict.values():
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in all_labels:  # Avoid duplicates
                    all_handles.append(h)
                    all_labels.append(l)
                    
            if coverage_handle is None and hasattr(ax, '_coverage_legend_handle'):
                coverage_handle = ax._coverage_legend_handle
                coverage_label = coverage_handle.get_label()

        if coverage_handle is not None and coverage_label is not None:
            if coverage_label not in all_labels:
                all_handles.append(coverage_handle)
                all_labels.append(coverage_label)

        # Place shared legend below the figure
        if all_handles:
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

        if axes is None and not return_figures:
            plt.show()
    
    if return_figures:
        return created_figures

    return None


def credit_dgp_tex_report(credit_dgp: CreditDataGenerator) -> str:
    r"""
    Generates a LaTeX-formatted report of the data generating process parameters.

    This method produces a comprehensive summary of the Gaussian mixture parameters
    for both the "bad" and "good" classes, including means, covariances, and
    mixture weights. The output is formatted as a LaTeX table for easy inclusion
    in academic papers or presentations.

    Returns:
        str: A string containing the LaTeX code for the report.
    """
    is_batched = credit_dgp.is_batched
    gb_list = ["bad", "good"]

    

    dgp_descr = [
        "The " + gb + f"s represent {(credit_dgp.bad_ratio if gb == 'bad' else 1 - credit_dgp.bad_ratio)*100:.1f}% of the data and its covariates' DGP is a "+ 
        (f"batched (B={getattr(credit_dgp, gb + '_mixture').B})" if is_batched else "") + 
        "Gaussian " + (
            "Mixture with " + ("deterministic" if credit_dgp.determinstic_mixture_weights else "random") + " weights"
            if getattr(credit_dgp, gb + "_mixture").is_mixture
            else "Distribution"
        ) + 
        " characterized by:\n\n"
        for gb in gb_list
    ]

    dist_descr = [
        getattr(credit_dgp, gb + "_mixture").univariate_mixture_as_tex(suffix = gb[0], letter_for_data='X', start_idx_mixtures=1)[1] # the dist_str
        for gb in gb_list
    ]

    report = (
        "The credit data DGP contains " + 
        (f"additive white noise with variance {credit_dgp.noise_std ** 2:.2f}" if credit_dgp.add_noise else "no noise") + 
        f" and samples with a bad ratio of {credit_dgp.bad_ratio*100:.1f}%."
    )

    bayes_error_rates_per: torch.Tensor = credit_dgp.bayes_error_rate(100_000) * 100

    if not is_batched:
        report += f"It's bayes error rate is {bayes_error_rates_per:.2f}%."

    report += " The " + " and ".join(gb_list) + " populations are defined as follows:\n\n"

    for dgp_d, dist_d in zip(dgp_descr, dist_descr):
            report += dgp_d 

            if not is_batched:
                report += dist_d[0] + "\n\n\n\n"
                continue
            
            for b_idx, dist_d_b in enumerate(dist_d):
                report += f"**Batch {b_idx}**: Bayes error rate is {bayes_error_rates_per[b_idx].item():.2f}%\n"
                report += dist_d_b + "\n\n"
            
    return report

def gaussian_mixture_tex_report(
        gm: GaussianMixture, 
        suffix: str, 
        letter_for_data: str = 'X', 
        start_idx_mixtures: int = 0,
        weights_iter_symbol: str = "k"
    ) -> Tuple[List[str], List[str]]:
        mus, covs_chol_decomp, weights = gm._normalized_params()
        covs = covs_chol_decomp @ covs_chol_decomp.mT
        is_mixture = gm.is_mixture
        is_batched = gm.is_batched

        tex_objs = []
        s = suffix

        def letter_maker(suffix, add_to_s = ""):
            full_subind = suffix + add_to_s
            if len(full_subind) > 0:
                return letter_for_data + "_{" + full_subind + r"}"
            
            return letter_for_data

        def mvn_latex(mu, cov, add_to_s = ""):
            mu_vec = (
                "\\begin{bmatrix}\n\t" +
                "\\\\\n\t".join([f"{mu_k:.1f}" for mu_k in mu]) +
                "\n\\end{bmatrix}"
            )
            Sigma_vcov = (
                "\\begin{bmatrix}\n\t" +
                "\\\\\n\t".join([" & ".join([f"{c:.1f}" for c in row]) for row in cov]) +
                "\n\\end{bmatrix}"
            )

            return (
                letter_maker(s, add_to_s) + r" \sim" + 
                r"\mathcal{N}\left(\mu_{" + s + add_to_s + "} = " + mu_vec +
                r", \Sigma_{" + s + add_to_s + "} = " + Sigma_vcov + r"\right)"
            )
        
        wi = weights_iter_symbol

        for b_idx, (mu_b, cov_b) in enumerate(zip(mus, covs)):
            if is_mixture:
                tex_objs.append([])
                current_weights = []
            for m_idx, (mu_m, cov_m) in enumerate(zip(mu_b, cov_b)):
                if is_mixture:
                    current_weights.append(weights[b_idx, m_idx] if is_batched else weights[0, m_idx])
                    m_idx += start_idx_mixtures
                    tex_objs[b_idx].append(mvn_latex(mu_m, cov_m, add_to_s=f"_{{{m_idx}}}"))
                else:
                    tex_objs.append(mvn_latex(mu_m, cov_m))
                    #break - not necessary, there will be only one iteration

            if is_mixture:
                subind = s + '_{' + wi + '}'
                tex_objs[b_idx].append(
                    letter_maker(s) + rf" \mid \{{Z = {wi} \}}  \sim \mathcal{{N}}\left(\mu_{{{subind}}}, \Sigma_{{{subind}}}\right)," +
                    r" \, Z \sim \text{{Categorical}}\left(" +
                        ", ".join([fr"\pi_{{{k}}}={w:.2f}" for k, w in zip(range(start_idx_mixtures, start_idx_mixtures+gm.K), current_weights)]) +
                    r"\right)"
                )

                
        dist_strs = []
        for text_obj in tex_objs:
            if is_mixture:
                vars_dist_str = "$$\n" + r", \;".join(text_obj[:-1]) + "\n$$"
                gm_dist_str = "$$\n" + text_obj[-1] + "\n$$"

                dist_strs.append(gm_dist_str + "\nwith\n" + vars_dist_str)
            else:
                dist_strs.append("$$\n" + text_obj + "\n$$")

        

        return text_obj, dist_strs