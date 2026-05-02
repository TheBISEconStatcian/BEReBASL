User: Considering the structure of a single diagnostics dictionary, create a function analogous to #sym:plot_lines_and_mains which for each diagnostic dictionary creates a plot with two axes, so nrow=2, a fig.suptitle filled with "Description" object as before. The axes should contain the following:

On the left I want a bar plot with very thin bars - if possible just vertical lines. The x-axis should be the sample_size. On each value of the sample size I want a bar for the acc_count and a bar for def_among_acc for each of the acc_name values. Always the acc_count should be left from the corresponding def_among_acc. To differentiate between them I want the acc_count to be solid and the  def_among_acc to be dashed. The color should differentiate the type of accepts. I want that the legend shows the type of accept by indicating the color and to also have in black a solid line that says "Accepts" and a dashed line that says "Defaults among accepts".

The second ax should contain the acc_rate and the def_rate_among_acc as line plots with the same logic as before: color differentiates the acc-type and solid vs dash differentiates between acc-rate and def-rate. For the legend use a logic like before.

Please let me know if you have any questions before starting to code. Remember to also generate a docstring when you code.

GitHub Copilot: I have a few clarifying questions before I start coding:

1. **Bar width and spacing**: For the bar plot on the left axis, should each sample_size point have a fixed number of bars (one pair per `acc_name`), or should the bars be very narrow/thin as you mentioned? How should bars for different `acc_name` values be positioned relative to each other at each sample_size point?

2. **Color palette**: Should I reuse the same color scheme from `plot_lines_and_mains` (from `plt.rcParams["axes.prop_cycle"]`), or do you want a different color mapping for the accept types?

3. **Sorting/ordering**: Should the `acc_name` values appear in a specific order along the x-axis at each sample_size point, or any consistent order is fine?

4. **Second y-axis**: Since `acc_rate` and `def_rate_among_acc` are rates (0-1 range) while `acc_count` and `def_among_acc` are counts (potentially much larger), should the left axis (bars) have its own y-axis scale independent of the line plot on the right axis?

5. **Legend placement**: For the legend, should it appear on both axes, or would you prefer separate legends for each subplot, or a single combined legend?

6. **Figure size**: What figure size would you like? Should I follow the same pattern as `plot_lines_and_mains` (e.g., `figsize=(16, 9)`)?

Please clarify these points and I'll create the function accordingly!

User: To your questions:

1. Yes each sample_size has a fixed number of bars: len(all_acc_vector_names)*len(["acc_count", "def_among_acc"]). And the bars should be positioned relative to each other. If you consider that a bar plot for 300 generation rounds is just too many bars, change the logic to line plots - that is also alright.

2. Yes please reuse the same color scheme and define at the beginning of the function once which colors goes with wich accepts to be unified across axes.

3. Any consistent order is fine, however the order in all_acc_vector_names would be not bad.

4. No it should not - the idea is exactly to give a feeling of the def_rate_among_acc while still being transparent on the counts - it should still be acceptable as per round around 300 new data points were generated

5. If it is no problem a single legend would be nicer

6. Figure size do it as you feel it is appropiate - it is a smaller plot than plot_lines_and_mains so a smaller figure size is probably appropiate

GitHub Copilot: ````python
def plot_diagnostics(
        diagnostics: Dict[str, Dict[str, torch.Tensor]],
        sample_sizes: torch.Tensor,
        sup_title: str = ""
    ) -> None:
    """
    Plot diagnostic metrics for acceptance decisions across generations.
    
    Creates a figure with two subplots:
    - Left (ax[0]): Line plot of acceptance counts and defaults among accepts per generation
    - Right (ax[1]): Line plot of acceptance rates and default rates among accepts per generation
    
    The visualization uses:
    - Colors to differentiate between different acceptance decision types (acc_names)
    - Solid lines for acceptance-related metrics (acc_count, acc_rate)
    - Dashed lines for default-related metrics (def_among_acc, def_rate_among_acc)
    
    Args:
        diagnostics: Dictionary mapping acceptance decision names to dictionaries of metrics.
                    Each metric is a tensor of shape (n_generations,) containing values per generation.
        sample_sizes: Tensor of shape (n_generations,) with the sample size at each generation.
        sup_title: Title for the figure (typically experiment description).
    """
    # Setup color mapping for accept types
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    acc_names = list(diagnostics.keys())
    color_map = {acc_name: to_rgb(colors[i % len(colors)]) for i, acc_name in enumerate(acc_names)}
    
    # Line styles: solid for counts/rates, dashed for defaults
    line_styles = {
        "acc_count": "-",
        "def_among_acc": "--",
        "acc_rate": "-",
        "def_rate_among_acc": "--"
    }
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    # Convert sample_sizes to numpy for plotting
    sample_sizes_np = sample_sizes.detach().cpu().numpy()
    generations = torch.arange(len(sample_sizes)).numpy()
    
    # ===== LEFT PLOT: Counts (acc_count, def_among_acc) =====
    ax_counts = axes[0]
    
    for acc_name in acc_names:
        acc_count = diagnostics[acc_name]["acc_count"].detach().cpu().numpy()
        def_among_acc = diagnostics[acc_name]["def_among_acc"].detach().cpu().numpy()
        
        color = color_map[acc_name]
        
        # Plot acceptance counts (solid line)
        ax_counts.plot(generations, acc_count, 
                      color=color, linestyle="-", linewidth=1.5, 
                      label=f"{acc_name} (Accepts)" if acc_name == acc_names[0] else "")
        
        # Plot defaults among accepts (dashed line)
        ax_counts.plot(generations, def_among_acc, 
                      color=color, linestyle="--", linewidth=1.5,
                      label=f"{acc_name} (Defaults)" if acc_name == acc_names[0] else "")
    
    ax_counts.set_xlabel("Generation")
    ax_counts.set_ylabel("Count")
    ax_counts.set_title("Acceptance and Default Counts per Generation")
    ax_counts.grid(True, alpha=0.3)
    
    # ===== RIGHT PLOT: Rates (acc_rate, def_rate_among_acc) =====
    ax_rates = axes[1]
    
    for acc_name in acc_names:
        acc_rate = diagnostics[acc_name]["acc_rate"].detach().cpu().numpy()
        def_rate_among_acc = diagnostics[acc_name]["def_rate_among_acc"].detach().cpu().numpy()
        
        color = color_map[acc_name]
        
        # Plot acceptance rates (solid line)
        ax_rates.plot(generations, acc_rate, 
                     color=color, linestyle="-", linewidth=1.5)
        
        # Plot default rates among accepts (dashed line)
        ax_rates.plot(generations, def_rate_among_acc, 
                     color=color, linestyle="--", linewidth=1.5)
    
    ax_rates.set_xlabel("Generation")
    ax_rates.set_ylabel("Rate")
    ax_rates.set_title("Acceptance and Default Rates per Generation")
    ax_rates.set_ylim([0, 1.05])
    ax_rates.grid(True, alpha=0.3)
    
    # ===== UNIFIED LEGEND =====
    # Create custom legend with both accept types and metrics
    legend_elements = []
    
    # Add accept type entries (colored lines)
    for acc_name in acc_names:
        legend_elements.append(
            plt.Line2D([0], [0], color=color_map[acc_name], linewidth=1.5, label=acc_name)
        )
    
    # Add metric type entries (black with different line styles)
    legend_elements.append(plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle="-", label="Accepts"))
    legend_elements.append(plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Defaults among Accepts"))
    
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9)
    
    if len(sup_title) > 0:
        fig.suptitle(sup_title, fontsize=14, y=1.00)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()


# Apply the function to all diagnostics
for sim_lbl, diagnostics in all_diagnostics.items():
    plot_diagnostics(
        diagnostics=diagnostics,
        sample_sizes=sim_objs[sim_lbl]["sample_sizes"],
        sup_title=sim_objs[sim_lbl]["Description"]
    )
````

User: That looks already quite good, however I do not like that the suptitle is too far away from the ax-boxes and that the legend of figure is too low from the boxes. Feels like too much white space wasted. How would you tune it so that it looks like I want? What should I change to reduce that white space?

GitHub Copilot: To reduce that extra white space:

- move the suptitle closer by raising its `y` value, e.g. `fig.suptitle(..., y=0.97)`
- shrink the reserved top/bottom margins in `tight_layout`, e.g. `fig.tight_layout(rect=[0, 0.05, 1, 0.95])`
- move the legend up slightly by changing its `bbox_to_anchor`, e.g. `bbox_to_anchor=(0.5, 0.02)`

So replace:

```
fig.legend(..., loc="upper center", bbox_to_anchor=(0.5, -0.02), ...)
if len(sup_title) > 0:
    fig.suptitle(sup_title, fontsize=14, y=1.00)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
```

with something like:

```
fig.legend(..., loc="upper center", bbox_to_anchor=(0.5, 0.02), ...)
if len(sup_title) > 0:
    fig.suptitle(sup_title, fontsize=14, y=0.97)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
```

If you want even tighter spacing, use `fig.subplots_adjust(top=0.92, bottom=0.09)` instead of `tight_layout`.

That will bring the title down and reduce the large gap below the axes.

User: Alright, now the functions give exactly what I want. Now I have a new mission for you. Adapting #sym:plot_diagnostics  and #sym:plot_lines_and_mains  I want you to create another loop that allows to make a HUGE plot containing the grid of the figures. You have seen how I read out of the keys the flip probability level and the spherical white noise's standard deviation. I want to have a huge plot where I can go to the right to see higher white noise standard deviations and downwards to see higher p(flip). This huge plots should be then saved each into it's own file that will be gigantic, but I want to be able to "navigate it" from my image visualization software. Let me know if you have any questions before coding. Consider that all combinations of white noise and flip are present, so that you can make a rectangular layout without having to handle "missing combinations"

GitHub Copilot: ````python
import os
import numpy as np
from matplotlib.lines import Line2D

def parse_sim_label_noise_flip(sim_lbl: str) -> tuple[float, float]:
    """
    Extract noise standard deviation and flip probability from sim label.
    Expected format: "noise_{noise_std}_flip_{flip_prob}".
    """
    flip_pos = sim_lbl.find("_flip")
    if flip_pos < 0:
        raise ValueError(f"Cannot parse flip probability from sim label: {sim_lbl}")
    noise_std = float(sim_lbl[len("noise_"):flip_pos].replace("_", "."))
    flip_prob = float(sim_lbl[flip_pos + len("_flip_"):].replace("_", "."))
    return noise_std, flip_prob


def plot_diagnostics_on_axes(
        diagnostics: Dict[str, Dict[str, torch.Tensor]],
        sample_sizes: torch.Tensor,
        ax_counts: plt.Axes,
        ax_rates: plt.Axes,
        title: str = ""
    ) -> None:
    """
    Plot diagnostics into a pair of provided axes.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    acc_names = list(diagnostics.keys())
    color_map = {acc_name: to_rgb(colors[i % len(colors)]) for i, acc_name in enumerate(acc_names)}

    sample_sizes_np = sample_sizes.detach().cpu().numpy()
    generations = torch.arange(len(sample_sizes)).numpy()

    for acc_name in acc_names:
        acc_count = diagnostics[acc_name]["acc_count"].detach().cpu().numpy()
        def_among_acc = diagnostics[acc_name]["def_among_acc"].detach().cpu().numpy()
        color = color_map[acc_name]

        ax_counts.plot(sample_sizes_np, acc_count, color=color, linestyle="-", linewidth=1.2)
        ax_counts.plot(sample_sizes_np, def_among_acc, color=color, linestyle="--", linewidth=1.2)

        acc_rate = diagnostics[acc_name]["acc_rate"].detach().cpu().numpy()
        def_rate_among_acc = diagnostics[acc_name]["def_rate_among_acc"].detach().cpu().numpy()
        ax_rates.plot(generations, acc_rate, color=color, linestyle="-", linewidth=1.2)
        ax_rates.plot(generations, def_rate_among_acc, color=color, linestyle="--", linewidth=1.2)

    ax_counts.set_xlabel("Generation")
    ax_counts.set_ylabel("Count")
    ax_counts.set_title(title or "Accepts / Defaults counts")
    ax_counts.grid(True, alpha=0.3)

    ax_rates.set_xlabel("Generation")
    ax_rates.set_ylabel("Rate")
    ax_rates.set_title("Accepts / Defaults rates")
    ax_rates.set_ylim([0, 1.05])
    ax_rates.grid(True, alpha=0.3)


def plot_diagnostics_grid(
        all_diagnostics: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        sim_objs: Dict[str, dict],
        output_path: str,
        cell_size: tuple[float, float] = (8, 4),
        dpi: int = 150
    ) -> None:
    """
    Create a grid figure with one diagnostics pair of plots per simulation.
    Columns increase with noise std, rows increase with flip probability.
    """
    sim_coords = {
        sim_lbl: parse_sim_label_noise_flip(sim_lbl)
        for sim_lbl in all_diagnostics
    }
    noise_vals = sorted({noise for noise, _ in sim_coords.values()})
    flip_vals = sorted({flip for _, flip in sim_coords.values()})

    n_rows = len(flip_vals)
    n_cols = len(noise_vals)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2 * n_cols,
        figsize=(2 * n_cols * cell_size[0], n_rows * cell_size[1]),
        dpi=dpi,
        squeeze=False
    )

    for sim_lbl, (noise_std, flip_prob) in sim_coords.items():
        row = flip_vals.index(flip_prob)
        col = noise_vals.index(noise_std)
        left_ax = axes[row, 2 * col]
        right_ax = axes[row, 2 * col + 1]
        title = f"noise={noise_std}, flip={flip_prob}"
        plot_diagnostics_on_axes(
            diagnostics=all_diagnostics[sim_lbl],
            sample_sizes=sim_objs[sim_lbl]["sample_sizes"],
            ax_counts=left_ax,
            ax_rates=right_ax,
            title=title
        )

    for row in range(n_rows):
        for col in range(2 * n_cols):
            axes[row, col].label_outer()

    legend_elements = []
    acc_names = list(next(iter(all_diagnostics.values())).keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        acc_name: to_rgb(colors[i % len(colors)])
        for i, acc_name in enumerate(acc_names)
    }
    for acc_name in acc_names:
        legend_elements.append(Line2D([0], [0], color=color_map[acc_name], lw=2, label=acc_name))
    legend_elements.append(Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="Accepts"))
    legend_elements.append(Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Defaults among Accepts"))

    fig.legend(handles=legend_elements, loc="lower center", ncol=min(len(legend_elements), 6), fontsize=10)
    fig.suptitle("Diagnostics grid: noise →, flip ↓", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_lines_and_mains_on_axes(
        line_collections_data: Dict[str, List],
        sample_sizes: torch.Tensor,
        axes_block: list[list[plt.Axes]]
    ) -> None:
    """
    Plot the 2x3 lines-and-mains block into the supplied axes block.
    """
    min_s, max_s = sample_sizes[[0, -1]]
    margin_x = (max_s - min_s) * 0.01

    dt_mapper = {
        "acc_based": "Accepts based",
        "oracle_naive": "Oracle naive",
        "oracle_comparable": "Oracle comparable"
    }
    metric_mapper = {
        "ks": "KS-statistic",
        "roc": "AU-ROC"
    }

    EXPECTATION_TYPES = ["acc_based", "oracle_naive", "oracle_comparable"]
    METRIC_CATEGORIES = ["ks", "roc"]

    for i_exp, exp in enumerate(EXPECTATION_TYPES):
        for i_m, m in enumerate(METRIC_CATEGORIES):
            ax = axes_block[i_m][i_exp]
            current_lbl = exp + "_" + m
            plot_dict = line_collections_data[current_lbl]

            ax.set_xlim(min_s - margin_x, max_s + margin_x)
            ax.set_ylim(0, 0.85)
            ax.add_collection(LineCollection(plot_dict["lines"], colors=plot_dict["colors"], linewidths=1.0))

            for main_dict in plot_dict["mains"]:
                ax.plot(*main_dict["args"], **main_dict["kwargs"])

            ax.set_ylabel(metric_mapper[m])
            ax.set_xlabel("Count generated data")
            ax.set_title(dt_mapper[exp])
            ax.autoscale()

    for row in axes_block:
        for ax in row:
            ax.label_outer()


def plot_lines_and_mains_grid(
        all_line_collections_data: Dict[str, Dict[str, List]],
        sim_objs: Dict[str, dict],
        output_path: str,
        cell_size: tuple[float, float] = (5, 4),
        dpi: int = 150
    ) -> None:
    """
    Create a massive grid figure composed of one 2x3 plot block per simulation.
    """
    sim_coords = {
        sim_lbl: parse_sim_label_noise_flip(sim_lbl)
        for sim_lbl in all_line_collections_data
    }
    noise_vals = sorted({noise for noise, _ in sim_coords.values()})
    flip_vals = sorted({flip for _, flip in sim_coords.values()})

    n_rows = len(flip_vals)
    n_cols = len(noise_vals)

    fig, axes = plt.subplots(
        nrows=2 * n_rows,
        ncols=3 * n_cols,
        figsize=(3 * n_cols * cell_size[0], 2 * n_rows * cell_size[1]),
        dpi=dpi,
        squeeze=False
    )

    for sim_lbl, (noise_std, flip_prob) in sim_coords.items():
        row_block = flip_vals.index(flip_prob)
        col_block = noise_vals.index(noise_std)

        block_axes = [
            axes[2 * row_block][3 * col_block:3 * col_block + 3],
            axes[2 * row_block + 1][3 * col_block:3 * col_block + 3]
        ]
        plot_lines_and_mains_on_axes(
            line_collections_data=all_line_collections_data[sim_lbl],
            sample_sizes=sim_objs[sim_lbl]["sample_sizes"],
            axes_block=block_axes
        )

    for i in range(2 * n_rows):
        for j in range(3 * n_cols):
            axes[i, j].label_outer()

    fig.suptitle("Lines and mains grid: noise →, flip ↓", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)

plot_diagnostics_grid(
    all_diagnostics=all_diagnostics,
    sim_objs=sim_objs,
    output_path=os.path.join(output_dir, "diagnostics_grid.png"),
    cell_size=(8, 4),
    dpi=150
)

plot_lines_and_mains_grid(
    all_line_collections_data=all_line_collections_data,
    sim_objs=sim_objs,
    output_path=os.path.join(output_dir, "lines_and_mains_grid.png"),
    cell_size=(5, 4),
    dpi=150
)
````

User: That does look almost perfect, however in the saved image generated by #sym:plot_lines_and_mains_grid  there is no legend. Could you add a legend at the bottom like in the case of #sym:plot_diagnostics_grid ? As you probably realized the label looks always the same in each suplot in my original plot_lines_and_mains - so one at the bottom of the huge figure would be enough

GitHub Copilot: ```python
def plot_lines_and_mains_grid(
        all_line_collections_data: Dict[str, Dict[str, List]],
        sim_objs: Dict[str, dict],
        output_path: str,
        cell_size: tuple[float, float] = (5, 4),
        dpi: int = 150
    ) -> None:
    """
    Create a massive grid figure composed of one 2x3 plot block per simulation.
    """
    sim_coords = {
        sim_lbl: parse_sim_label_noise_flip(sim_lbl)
        for sim_lbl in all_line_collections_data
    }
    noise_vals = sorted({noise for noise, _ in sim_coords.values()})
    flip_vals = sorted({flip for _, flip in sim_coords.values()})

    n_rows = len(flip_vals)
    n_cols = len(noise_vals)

    fig, axes = plt.subplots(
        nrows=2 * n_rows,
        ncols=3 * n_cols,
        figsize=(3 * n_cols * cell_size[0], 2 * n_rows * cell_size[1]),
        dpi=dpi,
        squeeze=False
    )

    for sim_lbl, (noise_std, flip_prob) in sim_coords.items():
        row_block = flip_vals.index(flip_prob)
        col_block = noise_vals.index(noise_std)

        block_axes = [
            axes[2 * row_block][3 * col_block:3 * col_block + 3],
            axes[2 * row_block + 1][3 * col_block:3 * col_block + 3]
        ]
        plot_lines_and_mains_on_axes(
            line_collections_data=all_line_collections_data[sim_lbl],
            sample_sizes=sim_objs[sim_lbl]["sample_sizes"],
            axes_block=block_axes,
            title_suffix=sim_objs[sim_lbl]["Description"]
        )

    for i in range(2 * n_rows):
        for j in range(3 * n_cols):
            axes[i, j].label_outer()

    legend_elements = []
    line_types = ["exp"] + REAL_PERFORMANCE_TYPES
    labels_linetypes = {
        "exp": "Expectation",
        "biased_acc": "Real. on Biased Accepts",
        "unbiased_acc": "Real. on Unbiased Accepts",
        "unbiased_future": "Real. on Unbiased Future Observations"
    }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {e: to_rgb(c) for e, c in zip(line_types, colors[:len(line_types)])}

    for lt in line_types:
        legend_elements.append(Line2D([0], [0], color=color_map[lt], lw=1.5, label=labels_linetypes[lt]))

    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), fontsize=10)

    fig.suptitle("Lines and mains grid: noise →, flip ↓", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
```