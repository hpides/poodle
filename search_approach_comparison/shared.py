"""
Shared constants, data loading, tokenisation, training, and evaluation helpers
used by run_baseline_search.py and run_proxy_search.py.
"""

import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.imdb.reduced_imdb import load_imdb_data

# ---------------------------------------------------------------------------
# Paths & registry
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_ROOT   = "/mount-fs/poodle/labeled-data/imdb/aclImdb"

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
BATCH_SIZE   = 32
MAX_LENGTH   = 256
LR           = 2e-5
NUM_EPOCHS   = 5
VAL_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_texts_and_labels(root, n_items, split):
    """Load exactly n_items examples from consecutive 500-item splits.

    n_items must be a positive multiple of 250.  When n_items is not a multiple
    of 500, the last partial chunk takes n_items % 500 examples evenly split
    between pos and neg (each split folder has equal pos/neg counts).

    Items within each loaded split are shuffled with seed 42 before any
    truncation so that partial chunks are always class-balanced.
    """
    assert n_items > 0 and n_items % 250 == 0, \
        f"n_items must be a positive multiple of 250, got {n_items}"
    texts, labels = [], []
    full_chunks = n_items // 500
    remainder   = n_items % 500   # 0 or 250

    for i in range(full_chunks + (1 if remainder else 0)):
        folder = os.path.join(root, f"{split}-500-{i}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected split directory not found: {folder}\n"
                f"DATA_ROOT is set to: {root}\n"
                f"Run prepare_data.py first if splits are missing."
            )
        t, l = load_imdb_data(folder)
        if not t:
            raise RuntimeError(f"No examples loaded from {folder}.")
        # Shuffle so pos/neg are interleaved before any truncation
        combined = list(zip(t, l))
        random.Random(42).shuffle(combined)
        t, l = zip(*combined)
        if i == full_chunks and remainder:
            # partial last chunk — balanced because shuffle interleaved pos/neg
            t = t[:remainder]
            l = l[:remainder]
        texts.extend(t)
        labels.extend(l)
    return texts, labels


# ---------------------------------------------------------------------------
# Tokenisation / dataset helpers
# ---------------------------------------------------------------------------
def tokenize(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True,
                     max_length=MAX_LENGTH, return_tensors="pt")


def make_dataset(tokenizer, texts, labels):
    enc = tokenize(tokenizer, texts)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if "token_type_ids" in enc:
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             enc["token_type_ids"], label_tensor)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], label_tensor)


def batch_to_model_inputs(batch, device, has_tti):
    if has_tti:
        input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in batch]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "token_type_ids": token_type_ids, "labels": labels}
    else:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, has_tti):
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
def evaluate(model, loader, device, has_tti):
    model.eval()
    correct = total = 0
    for batch in loader:
        inputs = batch_to_model_inputs(batch, device, has_tti)
        outputs = model(**inputs)
        preds  = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Full fine-tune (used by both baseline search and proxy search)
# ---------------------------------------------------------------------------
def full_fine_tune(ui_name, model_id, train_texts, train_labels,
                   test_texts, test_labels, device):
    """
    Fine-tune model_id on train_texts/labels for NUM_EPOCHS epochs,
    evaluate on test_texts/labels, return {'accuracy', 'processing_time_ms'}.
    Result is timed end-to-end (tokenizer load → test eval).
    """
    print(f"\n{'='*60}")
    print(f"  [FINE-TUNE] {ui_name}  |  {model_id}")
    print(f"{'='*60}")

    t_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    has_tti   = "token_type_ids" in tokenizer.model_input_names

    full_dataset = make_dataset(tokenizer, train_texts, train_labels)
    test_dataset = make_dataset(tokenizer, test_texts,  test_labels)

    n_val   = int(len(full_dataset) * VAL_FRACTION)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds,     batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,       batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_steps)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, has_tti)
        val_acc    = evaluate(model, val_loader, device, has_tti)
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    test_acc = evaluate(model, test_loader, device, has_tti)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    processing_time_ms = int((t_end - t_start) * 1000)
    print(f"  Test accuracy : {test_acc:.4f}  |  {processing_time_ms:,} ms")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"accuracy": round(test_acc, 4), "processing_time_ms": processing_time_ms}


# ---------------------------------------------------------------------------
# Proxy scoring (shared by run_proxy_search.py and run_proxy_sh_search.py)
# ---------------------------------------------------------------------------
PROXY_EPOCHS     = 100
PROXY_MAX_LENGTH = 512   # match model_search.py; fine-tuning uses MAX_LENGTH=256

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_search.nn_proxy import linear_proxy as _linear_proxy


@torch.no_grad()
def extract_cls_features(backbone, tokenizer, texts, device):
    """Extract pooler_output (or CLS token) features for a list of texts."""
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=PROXY_MAX_LENGTH, return_tensors="pt")
    has_tti = "token_type_ids" in enc
    all_features = []
    for i in range(0, len(texts), BATCH_SIZE):
        input_ids      = enc["input_ids"][i:i+BATCH_SIZE].to(device)
        attention_mask = enc["attention_mask"][i:i+BATCH_SIZE].to(device)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if has_tti:
            kwargs["token_type_ids"] = enc["token_type_ids"][i:i+BATCH_SIZE].to(device)
        outputs = backbone(**kwargs)
        feat = outputs.pooler_output if (
            hasattr(outputs, "pooler_output") and outputs.pooler_output is not None
        ) else outputs.last_hidden_state[:, 0, :]
        all_features.append(feat.cpu())
    return torch.cat(all_features, dim=0)


def run_proxy_score(ui_name, model_id, label,
                    train_texts, train_labels, test_texts, test_labels, device):
    """
    Proxy-score one model: freeze backbone, extract CLS features, train linear
    probe (100 epochs, SGD lr=0.1, seed 42).

    Returns {'proxy_score': float, 'processing_time_ms': int}.
    """
    print(f"\n  [PROXY-{label}] {ui_name}  |  {model_id}")

    t_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    backbone  = AutoModel.from_pretrained(model_id)
    backbone.to(device)
    backbone.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("    Extracting features …")
    train_features = extract_cls_features(backbone, tokenizer, train_texts, device)
    test_features  = extract_cls_features(backbone, tokenizer, test_texts,  device)

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("    Training linear probe …")
    torch.manual_seed(42)
    result = _linear_proxy(
        train_data=(train_features, torch.tensor(train_labels, dtype=torch.long)),
        test_data=(test_features,   torch.tensor(test_labels,  dtype=torch.long)),
        num_classes=2,
        device=device,
        batch_size=BATCH_SIZE,
        epochs=PROXY_EPOCHS,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    processing_time_ms = int((time.perf_counter() - t_start) * 1000)
    proxy_score = round(result["accuracy"], 4)

    print(f"    Proxy score: {proxy_score:.4f}  |  {processing_time_ms:,} ms")
    return {"proxy_score": proxy_score, "processing_time_ms": processing_time_ms}


# ---------------------------------------------------------------------------
# Pre-download helpers
# ---------------------------------------------------------------------------
def predownload_models(skip_if_cached=None):
    """
    Download all model weights into the HuggingFace cache before any timed work.
    Downloads both AutoTokenizer, AutoModel (backbone), and
    AutoModelForSequenceClassification so all variants are covered.

    skip_if_cached: optional callable(ui_name) -> bool. When it returns True
                    for a model, that model's download is skipped.
    """
    print("\nPre-downloading all models …")
    for ui_name, model_id in MODELS:
        if skip_if_cached and skip_if_cached(ui_name):
            print(f"  [SKIP] {ui_name} — already cached")
            continue
        print(f"  Downloading {ui_name} ({model_id}) …")
        tok  = AutoTokenizer.from_pretrained(model_id)
        bb   = AutoModel.from_pretrained(model_id)
        cls_ = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, ignore_mismatched_sizes=True)
        del tok, bb, cls_
    print("Pre-download complete.\n")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def require_gpu():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script must run on a GPU machine.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    return device
