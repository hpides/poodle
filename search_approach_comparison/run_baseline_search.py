"""
Baseline Search: fine-tune all 10 candidate models on IMDB (4000 train / 1000 test,
5 epochs) and record accuracy + wall-clock time per model.

Output: ../ui/baseline_search.json
Cache:  ./results/<ui_name>.json  (resumable — already-finished models are skipped)
"""

import json
import os
import sys


from shared import (
    DATA_ROOT, MODELS, RESULTS_DIR, SCRIPT_DIR,
    full_fine_tune, load_texts_and_labels, predownload_models, require_gpu,
)

UI_JSON_PATH     = os.path.join(SCRIPT_DIR, "..", "ui", "baseline_search.json")
NUM_TRAIN_ITEMS  = 4000
NUM_TEST_ITEMS   = 1000


def main():
    device = require_gpu()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data once
    print(f"\nLoading IMDB data from {DATA_ROOT} …")
    train_texts, train_labels = load_texts_and_labels(DATA_ROOT, NUM_TRAIN_ITEMS, "train")
    test_texts,  test_labels  = load_texts_and_labels(DATA_ROOT, NUM_TEST_ITEMS,  "test")
    print(f"  Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    predownload_models(
        skip_if_cached=lambda name: os.path.exists(os.path.join(RESULTS_DIR, f"{name}.json"))
    )

    # Fine-tune all models
    model_results = []
    for ui_name, model_id in MODELS:
        cache_path = os.path.join(RESULTS_DIR, f"{ui_name}.json")
        if os.path.exists(cache_path):
            print(f"\n[SKIP] {ui_name} — loading cached result")
            with open(cache_path) as f:
                result = json.load(f)
        else:
            result = full_fine_tune(
                ui_name, model_id,
                train_texts, train_labels,
                test_texts,  test_labels,
                device,
            )
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)

        model_results.append({
            "name":               ui_name,
            "accuracy":           result["accuracy"],
            "processing_time_ms": result["processing_time_ms"],
        })

    # Write UI JSON
    output = {
        "data_collection_items": 5000,
        "search_start_items":    5000,
        "labeling_speed":        13,
        "models":                model_results,
    }
    with open(UI_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Written {UI_JSON_PATH}")
    print(f"\n  {'Model':<10}  {'Accuracy':>10}  {'Time (ms)':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}")
    for m in model_results:
        print(f"  {m['name']:<10}  {m['accuracy']:>10.4f}  {m['processing_time_ms']:>12,}")


if __name__ == "__main__":
    main()
