import random

import matplotlib.pyplot as plt

# Real, publicly published per-1M-token prices (input/output), $.
# Sources:
#   - Together AI (Llama models): https://api.together.ai/models/...
#   - OpenAI (GPT models):        https://platform.openai.com/docs/pricing
#   - Google (Gemini models):     https://cloud.google.com/vertex-ai/generative-ai/docs/pricing
#   - Anthropic (Claude models):  https://www.anthropic.com/pricing
# "gpt-5.x" naming follows this repo's existing convention in
# cost_break_even/price_estimation.py (kept consistent with those numbers).
PRICING = {
    "Gemini 2.5 Flash-Lite": {"input": 0.10, "output": 0.40},
    "Phi-3-mini": {"input": 0.13, "output": 0.52},
    "Command R": {"input": 0.15, "output": 0.60},
    "Llama-8B": {"input": 0.20, "output": 0.20},
    "Mistral Small": {"input": 0.15, "output": 0.50},
    "GPT-5.4-nano": {"input": 0.20, "output": 1.25},
    "Mixtral-8x7B": {"input": 0.24, "output": 0.24},
    "Llama-13B": {"input": 0.30, "output": 0.30},
    "DeepSeek V3": {"input": 0.27, "output": 1.10},
    "GPT-4.1-mini": {"input": 0.40, "output": 1.60},
    "Gemini 2.5 Flash": {"input": 0.30, "output": 2.50},
    "Llama-34B": {"input": 0.60, "output": 0.60},
    "Llama-70B-turbo": {"input": 0.88, "output": 0.88},
    "Qwen-72B-turbo": {"input": 0.90, "output": 0.90},
    "Gemini 3 Flash (preview)": {"input": 0.50, "output": 3.00},
    "GPT-5.4-mini": {"input": 0.75, "output": 4.50},
    "Claude Haiku": {"input": 1.00, "output": 5.00},
    "Palmyra X4": {"input": 1.50, "output": 6.00},
    "Mistral Large": {"input": 2.00, "output": 6.00},
    "Jamba Large": {"input": 1.80, "output": 7.00},
    "GPT-4.1": {"input": 2.00, "output": 8.00},
    "Command R+": {"input": 2.20, "output": 9.00},
    "Grok": {"input": 2.50, "output": 10.00},
    "Llama-405B-turbo": {"input": 3.50, "output": 3.50},
    "Claude Sonnet": {"input": 3.00, "output": 15.00},
    "GPT-5.5": {"input": 5.00, "output": 30.00},
    "Claude Opus": {"input": 15.00, "output": 75.00},
}

# Plotted "cost" = mostly the input price, shifted a bit toward the (pricier)
# output price to roughly account for output tokens costing more, assuming
# most tokens in a request are input tokens.
OUTPUT_WEIGHT = 0.15

# Quality is a schematic, unitless 0-1 score (not a real benchmark number) -
# just placed so the GPT/Gemini models trace out the price/quality Pareto
# front with a clear step between each of them.
PARETO_QUALITY = {
    "Gemini 2.5 Flash-Lite": 0.40,
    "GPT-5.4-nano": 0.48,
    "GPT-4.1-mini": 0.56,
    "Gemini 2.5 Flash": 0.64,
    "Gemini 3 Flash (preview)": 0.72,
    "GPT-5.4-mini": 0.80,
    "GPT-4.1": 0.88,
    "GPT-5.5": 0.96,
}

# every other model is placed well below the front: its quality is the
# front's quality at (or just below) its own cost, minus a random gap, so
# they scatter around loosely instead of tracing a second, parallel line
RNG = random.Random(7)
MIN_GAP, MAX_GAP = 0.15, 0.55

COST = {
    name: price["input"] * (1 - OUTPUT_WEIGHT) + price["output"] * OUTPUT_WEIGHT
    for name, price in PRICING.items()
}

# on top of the named (real-priced) models above, throw in a batch of extra
# unnamed filler models with randomly sampled (log-uniform) cost, purely to
# populate the plot further - doubles the number of non-Pareto dots
N_EXTRA_MODELS = sum(1 for name in PRICING if name not in PARETO_QUALITY)
for i in range(N_EXTRA_MODELS):
    COST[f"other-model-{i}"] = 10 ** RNG.uniform(-0.7, 1.7)  # ~$0.2 - $50


def frontier_quality(cost):
    front = sorted((COST[name], q) for name, q in PARETO_QUALITY.items())
    best = front[0][1]
    for front_cost, front_quality in front:
        if front_cost > cost:
            break
        best = front_quality
    return best


QUALITY = dict(PARETO_QUALITY)
for name in COST:
    if name in PARETO_QUALITY:
        continue
    quality = frontier_quality(COST[name]) - RNG.uniform(MIN_GAP, MAX_GAP)
    QUALITY[name] = max(0.03, quality)

MIN_QUALITY = 0.3

# the GPT and Gemini models are the ones plotted on the price/quality Pareto
# front (best trade-off) - shown as solid dots with labels; everything else
# is shown faded out for context
PARETO_PREFIXES = ("GPT", "Gemini")

MODELS = {
    name: {"cost": COST[name], "quality": QUALITY[name]}
    for name in COST
    if QUALITY[name] >= MIN_QUALITY
}

# drop the priciest non-Pareto model so the axis doesn't have to stretch out
# just to fit one outlier - keeps the plot more compact
most_expensive_other = max(
    (n for n in MODELS if not n.startswith(PARETO_PREFIXES)),
    key=lambda n: MODELS[n]["cost"],
)
del MODELS[most_expensive_other]

COLOR = "#0868ac"


def plot_cost_vs_quality(models=MODELS, out_path="plots/cost_vs_quality"):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(8, 5.5))

    pareto = {n: p for n, p in models.items() if n.startswith(PARETO_PREFIXES)}
    other = {n: p for n, p in models.items() if not n.startswith(PARETO_PREFIXES)}

    ax.scatter(
        [p["cost"] for p in other.values()], [p["quality"] for p in other.values()],
        color=COLOR, alpha=0.2, s=100, zorder=2,
    )
    ax.scatter(
        [p["cost"] for p in pareto.values()], [p["quality"] for p in pareto.values()],
        color=COLOR, alpha=1.0, s=100, zorder=3,
    )

    # all labels use the same offset/direction (right next to the dot) so
    # there is no ambiguity about which label belongs to which point
    for name, point in pareto.items():
        ax.annotate(
            name,
            (point["cost"], point["quality"]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left", va="center",
            fontsize=15,
        )

    # symlog: linear from 0 up to linthresh, log beyond it - lets the axis
    # start at a real 0 while still reading as log scale for the data range.
    # linscale=1 makes the 0->linthresh segment the same width as one decade
    # in the log part (e.g. 0 -> $0.01 as wide as $0.1 -> $1)
    ax.set_xscale("symlog", linthresh=0.01, linscale=1)
    max_cost = max(p["cost"] for p in models.values())
    ax.set_xlim(0, max_cost * 1.3)
    ax.set_xticks([0, 0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0", "$0.01", "$0.1", "$1", "$10"])
    ax.set_xlabel("Cost (per 1M tokens, log scale)")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Quality")
    # schematic plot -> no numeric quality ticks
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # arrowheads at the end of the x/y axes (schematic style)
    ax.annotate(
        "", xy=(1, 0), xytext=(0.985, 0), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
        annotation_clip=False,
    )
    ax.annotate(
        "", xy=(0, 1), xytext=(0, 0.985), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
        annotation_clip=False,
    )

    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight")
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"Saved plot to {out_path}.png/.pdf")


if __name__ == "__main__":
    plot_cost_vs_quality()
