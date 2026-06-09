"""
Baseline Search: fine-tune all 10 candidate models on IMDB (4000 train / 1000 test)
and record accuracy + wall-clock training time per model.

Outputs: ../ui/baseline_search.json  (updates the UI demo file with real measurements)
Intermediate per-model results are cached in ./results/<ui_name>.json so the script
is resumable — already-finished models are skipped on rerun.

Training setup
--------------
- Optimizer  : AdamW, lr=2e-5  (appropriate for pretrained transformer fine-tuning)
- Scheduler  : linear decay, no warmup
- Epochs     : 5
- Batch size : 32
- Max length : 256 tokens
- Split      : 80 % train / 20 % validation (from the 4000 training examples)
- Device     : CUDA (script asserts GPU is available)
"""

import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
UI_JSON_PATH = os.path.join(SCRIPT_DIR, "..", "ui", "baseline_search.json")

# Raw IMDB data lives on the shared mount used by the rest of the project
DATA_ROOT = "/mount-fs/poodle/labeled-data/imdb/"

# ---------------------------------------------------------------------------
# Model registry  (model-N  →  HuggingFace ID)
# ---------------------------------------------------------------------------
MODELS = [
    ("model-1",  "google-bert/bert-base-uncased"),
    ("model-2",  "distilbert/distilbert-base-uncased-finetuned-sst-2-english"),
    ("model-3",  "rttl-ai/bert-base-uncased-yelp-reviews"),
    ("model-4",  "saitejautpala/bert-base-yelp-reviews"),
    ("model-5",  "Elenapervova/bio-distilbert-uncased-mimic-iii"),
    ("model-6",  "textattack/bert-base-uncased-CoLA"),
    ("model-7",  "cnut1648/biolinkbert-mednli"),
    ("model-8",  "Hate-speech-CNERG/bert-base-uncased-hatexplain"),
    ("model-9",  "JungleLee/bert-toxic-comment-classification"),
    ("model-10", "vittoriomaggio/bert-base-msmarco-fiqa"),
]

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_TRAIN_SPLITS = 8    # 8 × 500 = 4 000 training examples
NUM_TEST_SPLITS  = 2    # 2 × 500 = 1 000 test examples
NUM_EPOCHS       = 5
BATCH_SIZE       = 32
MAX_LENGTH       = 256
LR               = 2e-5
VAL_FRACTION     = 0.2  # 20 % of train used for validation


# ---------------------------------------------------------------------------
# Data loading  (reuse existing split layout)
# ---------------------------------------------------------------------------
def load_texts_and_labels(root, num_splits, split):
    """
    Load `num_splits` numbered sub-directories from root/<split>-500-{0..N-1}/
    Each directory contains pos/ and neg/ sub-folders with .txt review files.
    """
    texts, labels = [], []
    for i in range(num_splits):
        folder = os.path.join(root, f"{split}-500-{i}")
        for label_val, sentiment in [(1, "pos"), (0, "neg")]:
            sentiment_dir = os.path.join(folder, sentiment)
            if not os.path.isdir(sentiment_dir):
                raise FileNotFoundError(f"Missing directory: {sentiment_dir}")
            for fname in sorted(os.listdir(sentiment_dir)):
                if fname.endswith(".txt"):
                    with open(os.path.join(sentiment_dir, fname), encoding="utf-8") as f:
                        texts.append(f.read())
                    labels.append(label_val)
    return texts, labels


def make_dataset(tokenizer, texts, labels):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    # include token_type_ids only when the tokenizer provides them
    if "token_type_ids" in enc:
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             enc["token_type_ids"], label_tensor)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], label_tensor)


def batch_to_model_inputs(batch, device, has_token_type_ids: bool) -> dict:
    if has_token_type_ids:
        input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
    else:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, has_tti: bool):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = batch_to_model_inputs(batch, device, has_tti)
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, has_tti: bool) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        inputs = batch_to_model_inputs(batch, device, has_tti)
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-model fine-tuning
# ---------------------------------------------------------------------------
def fine_tune_model(
    ui_name: str,
    model_id: str,
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    device: torch.device,
) -> dict:
    """Fine-tune one model; return {'accuracy': float, 'processing_time_ms': int}."""

    print(f"\n{'='*60}")
    print(f"  {ui_name}  |  {model_id}")
    print(f"{'='*60}")

    # --- start clock (includes tokenizer load, model load, training) --------
    t_start = time.perf_counter()

    # --- tokenise -----------------------------------------------------------
    print("  Loading tokenizer & tokenising …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    has_tti = tokenizer.model_input_names.__contains__("token_type_ids")

    full_dataset = make_dataset(tokenizer, train_texts, train_labels)
    test_dataset = make_dataset(tokenizer, test_texts,  test_labels)

    # train / val split
    n_val   = int(len(full_dataset) * VAL_FRACTION)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds,   batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,     batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- model --------------------------------------------------------------
    print("  Loading model …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    # --- optimizer & scheduler ---------------------------------------------
    optimizer  = AdamW(model.parameters(), lr=LR)
    num_steps  = NUM_EPOCHS * len(train_loader)
    scheduler  = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_steps,
    )

    # --- training loop ------------------------------------------------------
    # Synchronise GPU so all preceding work (model.to(device)) is done
    # before we enter the first epoch.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, has_tti)
        val_acc    = evaluate(model, val_loader, device, has_tti)
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    # --- test evaluation ----------------------------------------------------
    test_acc = evaluate(model, test_loader, device, has_tti)

    # Stop clock after final evaluation; synchronise GPU so all work is done.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    processing_time_ms = int((t_end - t_start) * 1000)

    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Total time    : {processing_time_ms:,} ms  ({processing_time_ms/1000:.1f} s)")

    # free VRAM before loading the next model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy":            round(test_acc, 4),
        "processing_time_ms":  processing_time_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- GPU check ----------------------------------------------------------
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script must run on a GPU machine.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- load IMDB data once ------------------------------------------------
    print(f"\nLoading IMDB data from {DATA_ROOT} …")
    train_texts, train_labels = load_texts_and_labels(DATA_ROOT, NUM_TRAIN_SPLITS, "train")
    test_texts,  test_labels  = load_texts_and_labels(DATA_ROOT, NUM_TEST_SPLITS,  "test")
    print(f"  Train: {len(train_texts)} examples")
    print(f"  Test : {len(test_texts)} examples")

    # --- pre-download all models before any timed measurement ---------------
    # This ensures HuggingFace weights are cached locally so download time
    # does not pollute the fine-tuning measurements.
    print("\nPre-downloading all models …")
    for ui_name, model_id in MODELS:
        cache_path = os.path.join(RESULTS_DIR, f"{ui_name}.json")
        if os.path.exists(cache_path):
            print(f"  [SKIP] {ui_name} already has results, skipping download")
            continue
        print(f"  Downloading {ui_name} ({model_id}) …")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, ignore_mismatched_sizes=True
        )
        del tokenizer, model
    print("Pre-download complete.\n")

    # --- iterate over models ------------------------------------------------
    model_results = []

    for ui_name, model_id in MODELS:
        cache_path = os.path.join(RESULTS_DIR, f"{ui_name}.json")

        if os.path.exists(cache_path):
            print(f"\n[SKIP] {ui_name} — loading cached result from {cache_path}")
            with open(cache_path) as f:
                result = json.load(f)
        else:
            result = fine_tune_model(
                ui_name, model_id,
                train_texts, train_labels,
                test_texts,  test_labels,
                device,
            )
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {cache_path}")

        model_results.append({
            "name":                ui_name,
            "accuracy":            result["accuracy"],
            "processing_time_ms":  result["processing_time_ms"],
        })

    # --- write ui/baseline_search.json --------------------------------------
    output = {
        "data_collection_items": 5000,
        "search_start_items":    5000,
        "labeling_speed":        300,
        "models":                model_results,
    }

    with open(UI_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Written {UI_JSON_PATH}")
    print("\nSummary:")
    print(f"  {'Model':<10}  {'Accuracy':>10}  {'Time (ms)':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}")
    for m in model_results:
        print(f"  {m['name']:<10}  {m['accuracy']:>10.4f}  {m['processing_time_ms']:>12,}")


if __name__ == "__main__":
    main()
