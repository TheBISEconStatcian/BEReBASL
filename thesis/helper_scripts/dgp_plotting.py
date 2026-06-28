from matplotlib import pyplot as plt
from itertools import combinations
from math import log
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch

from typing import Dict, Optional, Union


from berebasl.simulation.credit_data_simulation import CreditDataGenerator

def extract_dgp_info_for_boundaries(
        dgp: CreditDataGenerator,
        var_to_hide: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, float]]:
    if var_to_hide is not None:
        var_to_hide = var_to_hide % dgp.F
        model_vars = [f for f in range(dgp.F) if f!= var_to_hide]
        cut_cov = dgp.bad_mixture.cov[..., model_vars, :][..., :, model_vars]
        inv_cov = torch.linalg.inv(cut_cov)
    else:
        model_vars = list(range(dgp.F))
        inv_cov = torch.cholesky_inverse(dgp.bad_mixture.cov_chol_decomp)
    mu_b = dgp.bad_mixture.mean[..., model_vars]
    mu_g = dgp.good_mixture.mean[..., model_vars]
    pi_b = dgp.prob_bad_given_no_shock

    # --- Precompute everything once ---
    W = inv_cov @ (mu_b - mu_g)

    beta = W @ ((mu_b + mu_g)/2)

    c = (1-pi_b)/pi_b

    if dgp.simulate_idiosyncratic_shocks:
        # --- Compute noisy shift ---
        pi_eps = 1 - dgp.prob_idiosyncratic_shock
        pi_vareps_b = dgp.prob_bad_given_shock

        xi = (0.5 - (1-pi_eps)*pi_vareps_b) / pi_eps
        c *= xi / (1-xi)

    bias = beta + log(c)   # final intercept

    return {
        "W" : W,
        "bias" : bias,
        "mu_b" : mu_b,
        "mu_g" : mu_g,
        "pi_b" : pi_b
    }

def plot3d_with_perf_bayes_boundary(
        dgp: CreditDataGenerator,
        W: torch.Tensor,
        bias: torch.Tensor,
        title: str = ""
    ):

    fig = plt.figure(figsize=(12,4.5))

    ax = fig.add_subplot(1,1, 1, projection="3d")

    ellipsoid_probs = (.8,)
    dgp.plot_credit_dgp_3d(
        sample_size=10_000,
        axes=[ax], elev=15, azim=20,
        ellipsoid_probs=ellipsoid_probs
    )
    #pos = ax.get_position()
    ax.set_box_aspect((4, 4, 3))
    ax.set_xlim(-6, 6)
    ax.set_zlim(-6, 6)
    ax.set_ylim(-6, 6)

    # --- Compute LDA direction ---
    # --- Create grid for plane ---
    xx, zz = torch.meshgrid(
            torch.linspace(-6, 6, 500),
            torch.linspace(-6, 6, 500),
            indexing="xy"
        )

    # Solve for z: w1*x + w2*y + w3*z + b = 0
    yy = (bias - W[0]*xx - W[2]*zz) / W[1]

    # Convert to numpy for matplotlib
    xx_np = xx.numpy()
    yy_np = yy.numpy()
    zz_np = zz.numpy()

    # --- Plot the plane ---
    surf = ax.plot_surface(xx_np, yy_np, zz_np, alpha=0.3, color="black", label="Perf. Bayes boundary")

    # --- Create a proxy artist for the legend ---
    # We match the facecolor (black) and alpha (0.3) of the surface

    # --- Append to existing legend handles ---
    # 1. Grab what is already in the legend (from dgp_base plot)
    handles, labels = [], []
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

    # 2. Append your new proxy artist and its label
    #handles.append(surface_proxy)
    #labels.append("Perf. Bayes boundary")
    #fig.legends.append()

    fig.legend(
        handles=[surf],
        labels=["$\\mathcal{D}^*$"],
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=max(1, min(len(handles), 4)),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        columnspacing=1.2
    )

    if len(title) > 0:
        ax.set_title(title)


    ax.set_xlabel("$X_{m, 1}$")
    ax.set_ylabel("$X_{m, 2}$")
    ax.set_zlabel("$X_{h}$")

    ax.set_zlabel("$X_h$")
    zlab = ax.zaxis.get_label()
    zlab.set_rotation(90) 

    pos = ax.get_position()
    ax.set_position([pos.x0 - 0.05, pos.y0 - 0.05, pos.width + 0.12, pos.height])


    plt.show()

def plot_pairwise_dgp_with_perf_bayes_boundaries(
        dgp: CreditDataGenerator,
        W: torch.Tensor,
        bias: torch.Tensor,
        mu_b: torch.Tensor,
        mu_g: torch.Tensor,
        pi_b: float,
        fig_title: str = ""
) -> None:
    fig = plt.figure(figsize=(12,4))
    F = dgp.F
    aes_dict = {}
    lbl_maker = lambda f: f"$X_{{m, {f+1}}}$" if f < F-1 else "$X_{h}$"
    for (f1, f2), a_idx in zip(combinations(range(F), 2), range(1, 4)):
        aes_dict[f"{f1:02}{f2:02}"] = fig.add_subplot(1,3, a_idx)
        aes_dict[f"{f1:02}{f2:02}"].set_xlabel(lbl_maker(f1))
        aes_dict[f"{f1:02}{f2:02}"].set_ylabel(lbl_maker(f2))

    _, handles_2d, legends_2d = dgp.pairwise_plot_dgp(
        sample_size=18_000, axes=[aes_dict]
    )

    # --- Add decision boundary to each 2D subplot ---
    # --- Add decision boundary to each 2D subplot ---
    f_all = set(range(F))

    for (f1, f2), a_idx in zip(combinations(range(F), 2), range(1, 4)):
        ax = aes_dict[f"{f1:02}{f2:02}"]

        # pick the remaining coordinate index
        f3 = list(f_all - {f1, f2})[0]

        # grid for x_f1
        x2_vals = torch.linspace(-6, 6, 500) #if f1==1 and f2==2 else torch.linspace(*ax.get_xlim(), 200)

        # FIXED VALUE FOR THE UNPLOTTED COORDINATE
        x3_fixed = (1 - pi_b) * mu_g[f3] + pi_b * mu_b[f3]

        # compute x_f2 from plane equation
        x1_vals = (bias - W[f2] * x2_vals - W[f3] * x3_fixed) / W[f1]

        line_proj_perf, = ax.plot(
            x1_vals.numpy(),
            x2_vals.numpy(),
            color="darkgray",
            linestyle="--",
            linewidth=1.5
        )

        constr_dgp_info = extract_dgp_info_for_boundaries(dgp, var_to_hide = f3)
        f2_greater_f1 = f2 > f1
        idx_constr_space_f1 = int(not f2_greater_f1)
        idx_constr_space_f2 = int(f2_greater_f1)
        W_constr, b_constr = constr_dgp_info["W"], constr_dgp_info["bias"]
        x1_vals_constr = (b_constr - W_constr[idx_constr_space_f2] * x2_vals) / W_constr[idx_constr_space_f1]

        line_proj_constr, = ax.plot(
            x1_vals_constr.numpy(),
            x2_vals.numpy(),
            color="darkred",
            linestyle=":",
            linewidth=1.5
        )

        ax.set_ylim(-6,6)
        ax.set_xlim(-6,6)


    fig.legend(
        handles_2d + [line_proj_perf, line_proj_constr],
        legends_2d + [
            "$\\mathcal{D}^* | X' = \\mathbb{E}(X')$",
            "$\\mathcal{D}^r$"
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(handles_2d) + 2,
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        columnspacing=1.2
    )
    
    if len(fig_title) > 0:
        fig.suptitle(fig_title, fontsize=16)

    fig.tight_layout()


    plt.show()