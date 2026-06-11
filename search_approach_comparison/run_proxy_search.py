"""
Proxy Scoring: rank all 10 models via a linear probe on frozen CLS features,
then fully fine-tune only the winning model.

Two variants (differ only in proxy data size):
  small — 400 train / 100 test  →  ../ui/proxy_scoring_500.json
  large — 4000 train / 1000 test →  ../ui/proxy_scoring_5000.json

Step 1 — Proxy (all 10 models): freeze transformer, extract CLS features,
          train linear head 100 epochs (SGD lr=0.1), record score + time.
Step 2 — Full fine-tune (winner only, 4000/1000): reuses run_baseline_search.py
          cache if available, otherwise runs from scratch.

Cache:
  ./results/proxy_<variant>_<ui_name>.json  — proxy results per variant
  ./results/<ui_name>.json                  — fine-tune results (shared)
"""

import json
import os
import sys
import time

from shared import (
    DATA_ROOT, MODELS, RESULTS_DIR, SCRIPT_DIR,
    full_fine_tune, load_texts_and_labels, predownload_models, require_gpu,
    run_proxy_score,
)

# ---------------------------------------------------------------------------
# Proxy variants
# ---------------------------------------------------------------------------
PROXY_VARIANTS = [
    {
        "name":            "small",
        "proxy_train":     500,
        "proxy_test":      500,
        "ft_train":        500,
        "ft_test":         500,
        "ui_json":         os.path.join(SCRIPT_DIR, "..", "ui", "proxy_scoring_500.json"),
        "data_collection": 500,
        "search_start":    500,
    },
]


# ---------------------------------------------------------------------------
# Run one variant end-to-end
# ---------------------------------------------------------------------------
def run_variant(variant, device):
    name = variant["name"]
    print(f"\n{'#'*60}")
    print(f"  VARIANT: {name}  (proxy train={variant['proxy_train']} / test={variant['proxy_test']})")
    print(f"{'#'*60}")

    proxy_train_texts, proxy_train_labels = load_texts_and_labels(
        DATA_ROOT, variant["proxy_train"], "train")
    proxy_test_texts, proxy_test_labels = load_texts_and_labels(
        DATA_ROOT, variant["proxy_test"], "test")
    print(f"  Proxy data     — train: {len(proxy_train_texts)}  test: {len(proxy_test_texts)}")

    ft_train_texts, ft_train_labels = load_texts_and_labels(DATA_ROOT, variant["ft_train"], "train")
    ft_test_texts,  ft_test_labels  = load_texts_and_labels(DATA_ROOT, variant["ft_test"],  "test")
    print(f"  Fine-tune data — train: {len(ft_train_texts)}  test: {len(ft_test_texts)}")

    # Proxy score all models
    proxy_results = []
    for ui_name, model_id in MODELS:
        cache_path = os.path.join(RESULTS_DIR, f"proxy_{name}_{ui_name}.json")
        if os.path.exists(cache_path):
            print(f"\n  [SKIP] proxy-{name} {ui_name} — loading cache")
            with open(cache_path) as f:
                result = json.load(f)
        else:
            result = run_proxy_score(
                ui_name, model_id, name.upper(),
                proxy_train_texts, proxy_train_labels,
                proxy_test_texts,  proxy_test_labels,
                device,
            )
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)

        proxy_results.append({
            "name":               ui_name,
            "model_id":           model_id,
            "proxy_score":        result["proxy_score"],
            "processing_time_ms": result["processing_time_ms"],
        })

    # Full fine-tune winner
    winner = max(proxy_results, key=lambda m: m["proxy_score"])
    print(f"\n  Winner ({name}): {winner['name']} (proxy score {winner['proxy_score']:.4f})")

    ft_cache = os.path.join(RESULTS_DIR, f"ft_{name}_{winner['name']}.json")
    if os.path.exists(ft_cache):
        print(f"  [SKIP] Full fine-tune — loading cached result")
        with open(ft_cache) as f:
            ft_result = json.load(f)
    else:
        ft_result = full_fine_tune(
            winner["name"], winner["model_id"],
            ft_train_texts, ft_train_labels,
            ft_test_texts,  ft_test_labels,
            device,
        )
        with open(ft_cache, "w") as f:
            json.dump(ft_result, f, indent=2)

    # Write output JSON
    output_models = []
    for m in proxy_results:
        entry = {
            "name":               m["name"],
            "accuracy":           m["proxy_score"],
            "processing_time_ms": m["processing_time_ms"],
        }
        if m["name"] == winner["name"]:
            entry["full_accuracy"]           = ft_result["accuracy"]
            entry["full_processing_time_ms"] = ft_result["processing_time_ms"]
        output_models.append(entry)

    output = {
        "data_collection_items": variant["data_collection"],
        "search_start_items":    variant["search_start"],
        "labeling_speed":        13,
        "models":                output_models,
    }
    with open(variant["ui_json"], "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Written {variant['ui_json']}")

    # Write the winner's ft result to ui/ft_<variant>_<winner>.json
    # so the UI can load it directly for the fine-tuning animation.
    ui_dir = os.path.join(SCRIPT_DIR, "..", "ui")
    ft_ui_path = os.path.join(ui_dir, f"ft_{name}_{winner['name']}.json")
    with open(ft_ui_path, "w") as f:
        json.dump(ft_result, f, indent=2)
    print(f"  ✓ Written {ft_ui_path}")

    print(f"\n  {'Model':<10}  {'Proxy score':>12}  {'Time (ms)':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}")
    for m in sorted(proxy_results, key=lambda x: x["proxy_score"], reverse=True):
        tag = " ← winner" if m["name"] == winner["name"] else ""
        print(f"  {m['name']:<10}  {m['proxy_score']:>12.4f}  {m['processing_time_ms']:>12,}{tag}")
    print(f"\n  Winner fine-tune: accuracy={ft_result['accuracy']:.4f}  "
          f"time={ft_result['processing_time_ms']:,} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = require_gpu()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    predownload_models(
        skip_if_cached=lambda name: all(
            os.path.exists(os.path.join(RESULTS_DIR, f"proxy_{v['name']}_{name}.json"))
            for v in PROXY_VARIANTS
        )
    )

    for variant in PROXY_VARIANTS:
        run_variant(variant, device)

    print("\n✓ All variants complete.")


if __name__ == "__main__":
    main()
