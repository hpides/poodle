"""
Proxy Scoring with Successive Halving (SH):

  Round 1 — all 10 models, 250 train / 250 test  → keep top 5
  Round 2 — top 5 models,  500 train / 500 test  → keep top 3
  Round 3 — top 2 models, 1000 train / 1000 test → winner

  Full fine-tune — winner on 4000 train / 1000 test (reuses baseline search cache)

Output : ../ui/proxy_scoring_sh.json  (overwrites the dummy file with real measurements)
Cache  : ./results/sh_r<N>_<ui_name>.json  per round
         ./results/ft_sh_<ui_name>.json    fine-tune winner
"""

import json
import os
import sys
import time

import torch

from shared import (
    DATA_ROOT, MODELS, RESULTS_DIR, SCRIPT_DIR,
    load_texts_and_labels, predownload_models, require_gpu,
    run_proxy_score,
)

# ---------------------------------------------------------------------------
# SH configuration
# ---------------------------------------------------------------------------
ROUNDS = [
    {"round": 1, "train": 250,  "test": 250,  "keep": 5},
    {"round": 2, "train": 500,  "test": 500,  "keep": 3},
    {"round": 3, "train": 1000, "test": 1000, "keep": 1},  # keep=1 → winner
]

BASELINE_SEARCH_JSON = os.path.join(SCRIPT_DIR, "..", "ui", "baseline_search.json")
UI_JSON_PATH         = os.path.join(SCRIPT_DIR, "..", "ui", "proxy_scoring_sh.json")


# ---------------------------------------------------------------------------
# Main SH loop
# ---------------------------------------------------------------------------
def run_sh(device):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Candidates start as the full model list
    candidates = list(MODELS)   # [(ui_name, model_id), ...]
    rounds_output = []          # collected for the output JSON

    for rnd in ROUNDS:
        r       = rnd["round"]
        n_train = rnd["train"]
        n_test  = rnd["test"]
        keep    = rnd["keep"]

        print(f"\n{'='*60}")
        print(f"  SH ROUND {r}  —  {len(candidates)} models  "
              f"|  {n_train} train / {n_test} test  |  keep top {keep}")
        print(f"{'='*60}")

        # Load data for this round (each round uses independent splits starting at index 0)
        train_texts, train_labels = load_texts_and_labels(DATA_ROOT, n_train, "train")
        test_texts,  test_labels  = load_texts_and_labels(DATA_ROOT, n_test,  "test")

        round_results = []
        for ui_name, model_id in candidates:
            cache_path = os.path.join(RESULTS_DIR, f"sh_r{r}_{ui_name}.json")
            if os.path.exists(cache_path):
                print(f"\n  [SKIP] R{r} {ui_name} — loading cache")
                with open(cache_path) as f:
                    result = json.load(f)
            else:
                result = run_proxy_score(
                    ui_name, model_id, f"SH-R{r}",
                    train_texts, train_labels,
                    test_texts,  test_labels,
                    device,
                )
                with open(cache_path, "w") as f:
                    json.dump(result, f, indent=2)

            round_results.append({
                "name":               ui_name,
                "model_id":           model_id,
                "proxy_score":        result["proxy_score"],
                "processing_time_ms": result["processing_time_ms"],
            })

        # Sort by proxy score descending
        round_results.sort(key=lambda m: m["proxy_score"], reverse=True)

        print(f"\n  Round {r} ranking:")
        for i, m in enumerate(round_results):
            tag = " ✓ keep" if i < keep else " ✗ drop"
            print(f"    {m['name']:<10}  {m['proxy_score']:.4f}{tag}")

        rounds_output.append({
            "round":      r,
            "data_items": n_train,
            "models": [
                {"name": m["name"], "proxy_score": m["proxy_score"]}
                for m in round_results
            ],
        })

        # Prune to top-K for the next round
        candidates = [(m["name"], m["model_id"]) for m in round_results[:keep]]

    # ---------------------------------------------------------------------------
    # Look up winner's fine-tune result from baseline_search.json
    # ---------------------------------------------------------------------------
    winner_name, _ = candidates[0]
    print(f"\n{'='*60}")
    print(f"  Winner: {winner_name} — looking up fine-tune result from baseline_search.json")
    print(f"{'='*60}")

    with open(BASELINE_SEARCH_JSON) as f:
        baseline_data = json.load(f)
    baseline_by_name = {m["name"]: m for m in baseline_data["models"]}
    if winner_name not in baseline_by_name:
        raise KeyError(f"Winner '{winner_name}' not found in {BASELINE_SEARCH_JSON}")
    bl = baseline_by_name[winner_name]
    ft_result = {
        "accuracy":           bl["accuracy"],
        "processing_time_ms": bl["processing_time_ms"],
    }

    print(f"\n  Fine-tune accuracy: {ft_result['accuracy']:.4f}  "
          f"({ft_result['processing_time_ms']:,} ms)")

    # ---------------------------------------------------------------------------
    # Write proxy_scoring_sh.json
    # ---------------------------------------------------------------------------
    # Compute ms_per_data_item from round-1 timings (avg across all models)
    r1_results = []
    for ui_name, _ in MODELS:
        cp = os.path.join(RESULTS_DIR, f"sh_r1_{ui_name}.json")
        if os.path.exists(cp):
            with open(cp) as f:
                r1_results.append(json.load(f))
    if r1_results:
        avg_ms = sum(r["processing_time_ms"] for r in r1_results) / len(r1_results)
        ms_per_data_item = round(avg_ms / ROUNDS[0]["train"], 1)
    else:
        ms_per_data_item = 1

    output = {
        "data_collection_items": 5000,
        "search_start_items":    1000,
        "labeling_speed":        13,
        "ms_per_data_item":      ms_per_data_item,
        "winner": {
            "name":               winner_name,
            "accuracy":           ft_result["accuracy"],
            "processing_time_ms": ft_result["processing_time_ms"],
        },
        "rounds": rounds_output,
    }

    with open(UI_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Written {UI_JSON_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = require_gpu()

    # Pre-download all models before any timed work
    predownload_models(
        skip_if_cached=lambda name: all(
            os.path.exists(os.path.join(RESULTS_DIR, f"sh_r{rnd['round']}_{name}.json"))
            for rnd in ROUNDS
        )
    )

    run_sh(device)
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
