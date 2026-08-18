import argparse
from typing import List, Dict

import matplotlib.pyplot as plt


# Baseline:
# We need to use 5000 items for training; below, we do not get the target accuracy.
# 1) Label 5000 items using LLM: 5000items / 13items/s                                                ; 385 s
# 2) Search Model: Just use BERT default model                                                        ;   0 s
# 3) Fine-tune model using 5000 items: 5 epochs * (74sec-18s) (fine-tune - inf of 4500 items)         ; 280 s
# SUMME                                                                                               ; 665 s
# Result 0.89 accuracy
#
# Baseline + Naive Model Search:
# We need to use 5000 items for training; below, we do not get the target accuracy.
# 1) Label 5000 items using LLM: 5000items / 13items/s                                                ; 385 s
# 2) Search Model: Fine-tune all 10 models                                                            ;2800 s
# 3) Just select best fine-tuned model                                                                ;   0 s
# SUMME                                                                                               ;3185 s
# Result 0.92 accuracy
#
# Model Search (Fine-tune on 500 items)
# 1) Label 1000 items using LLM: 500items / 13items/s                                                 ;  38 s
# 2) Search Model: 10 models * 6s/model                                                               ;  60 s
# 3) Fine-tune model using 500 items: 10 epochs * 7sec (fine-tune)                                    ;  70 s
# SUMME                                                                                               ; 168 s (3.96 Base, 18.96 Base + Naive Model Search)
# Result 0.91 accuracy
#
# Model Search (Fine-tune on 5000 items)
# 1) Label 5000 items using LLM: 500items / 13items/s                                                 ; 385 s
# 2) Search Model: 10 models * 6s/model                                                               ;  60 s
# 3) Fine-tune model using 5000 items: 5 epochs * (74sec-18s) (fine-tune - inf of 4500 items)         ; 280 s
# SUMME                                                                                               ; 725 s (4.39 Base + Naive Model Search)
# Result 0.92 accuracy
#

colors = ['#bae4bc', '#43a2ca', '#0868ac']


def sample_data():
    # Example times in seconds for a few approaches/models
    return [
        {"approach": "Baseline", "label_seconds": 385, "search_seconds": 0, "fine_tune_seconds": 280, "accuracy": 0.89},
        {"approach": "S-naive", "label_seconds": 385, "search_seconds": 2800, "fine_tune_seconds": 0, "accuracy": 0.92},
        {"approach": "S-5000", "label_seconds": 385, "search_seconds": 60, "fine_tune_seconds": 280, "accuracy": 0.92},
        {"approach": "S-500", "label_seconds": 38, "search_seconds": 60, "fine_tune_seconds": 70, "accuracy": 0.91},
    ]


def format_seconds(s: float) -> str:
    if s is None:
        return "0s"
    s = float(s)
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    if s >= 1:
        return f"{s:.0f}s"
    return f"{s * 1000:.0f}ms"



def plot_stack(rows: List[Dict], out_path: str = "timings.png"):
    if not rows:
        print("No data to plot")
        return

    approaches = [r["approach"] for r in rows]
    labels = [r["label_seconds"] for r in rows]
    searches = [r["search_seconds"] for r in rows]
    fines = [r["fine_tune_seconds"] for r in rows]
    accuracies = [r.get("accuracy", None) for r in rows]

    totals = [labels[i] + searches[i] + fines[i] for i in range(len(rows))]

    # convert seconds -> minutes for plotting
    labels_min = [s / 60.0 for s in labels]
    searches_min = [s / 60.0 for s in searches]
    fines_min = [s / 60.0 for s in fines]
    totals_min = [labels_min[i] + searches_min[i] + fines_min[i] for i in range(len(rows))]

    x = range(len(approaches))

    # Create a broken y-axis: bottom shows 0-15 minutes, top shows 40 -> max
    # make figure much shorter so the plotted axes occupy only a small fraction
    # of the vertical space (plot area appears "thin" — roughly 1/3 of total height)
    # keep some relative difference between top/bottom subplots, but keep them compact
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True,
        figsize=(5,2),  # width, height in inches
        gridspec_kw={"height_ratios": [0.4, 1]}
    )

    # Plot the same stacked bars on both axes; clipping will hide out-of-range parts
    p1b = ax_bottom.bar(x, labels_min, label="Label items", color=colors[0])
    p2b = ax_bottom.bar(x, searches_min, bottom=labels_min, label="Model search", color=colors[1])
    bottom_for_fines = [labels_min[i] + searches_min[i] for i in range(len(labels_min))]
    p3b = ax_bottom.bar(x, fines_min, bottom=bottom_for_fines, label="Fine-tuning", color=colors[2])

    p1t = ax_top.bar(x, labels_min, color=colors[0])
    p2t = ax_top.bar(x, searches_min, bottom=labels_min, color=colors[1])
    p3t = ax_top.bar(x, fines_min, bottom=bottom_for_fines, color=colors[2])

    # removed leftover `ax` reference (use ax_bottom / ax_top instead)

    # ensure bottom axis shows ticks/labels (0-15 plot)
    ax_bottom.set_xticks(list(x))
    ax_bottom.set_xticklabels(approaches)
    # explicitly enable bottom ticks/labels to avoid them being hidden by shared-x behavior
    ax_bottom.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)

    # top subplot: do not show x tick labels or tick marks (do not clear shared ticks)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)

    # removed fixed-position fig.text; we will place the joint y-label after layout

    # remove secondary axis; instead annotate accuracy above each stacked bar
    # use explicit legend handles ordered to match the stacked bars (top -> bottom)
    label_handle = p1b[0]
    search_handle = p2b[0]
    fine_handle = p3b[0]
    # place legend immediately to the right of the top plot (minimal gap)
    ax_top.legend(
        [fine_handle, search_handle, label_handle],
        ["Fine-tuning", "Model search", "Label items"],
        loc="center left",
        # lowered the vertical anchor so the legend sits a bit down
        bbox_to_anchor=(1.005, 0),
        bbox_transform=ax_top.transAxes,
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=0.6,
    )

    # Add a small headroom above the highest bar so title/text doesn't overlap
    max_total = max(totals_min) if totals_min else 0.0
    headroom = max_total * 0.12 if max_total > 0 else 0.1
    # Configure broken y-limits:
    low_top = 12.0
    high_bottom = 45.0
    top_ylim_upper = 55
    ax_bottom.set_ylim(0, low_top)
    ax_top.set_ylim(high_bottom, top_ylim_upper)

    # Hide the spines between the two axes and add diagonal break marks
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)  # no duplicate top labels
    d = .010  # slightly smaller diagonal markers (works better with reduced gap)
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)        # bottom-left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # annotate accuracy (as percentage) above each bar
    # accuracy annotations removed — no text labels above bars

    plt.tight_layout()
    # reduce vertical gap between the two subplots (small positive value)
    fig.subplots_adjust(hspace=0.1)
    # ensure final positions are computed, then compute vertical midpoint between axes
    fig.canvas.draw()
    pos_top = ax_top.get_position()
    pos_bottom = ax_bottom.get_position()
    # midpoint between the bottom of the top axis and the top of the bottom axis
    mid_y = (pos_top.y0 + pos_bottom.y1) / 2.0
    # move label slightly further left to clear ticks and a bit down from the midpoint
    x_pos = -0.008
    y_pos = mid_y - 0.1
    # clamp into figure bounds
    y_pos = max(0.03, min(0.97, y_pos))
    fig.text(x_pos, y_pos, "Time (minutes)", va="center", ha="left", rotation="vertical", fontsize=10)
    plt.savefig(f"{out_path}.png", bbox_inches="tight")
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot stacked timing bar chart")
    parser.add_argument("--out", default="./plots/timings", help="Output image path (PNG)")
    args = parser.parse_args()

    rows = sample_data()

    plot_stack(rows, args.out)


if __name__ == "__main__":
    main()
