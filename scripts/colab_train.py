"""Fine-tune a small transformer on the complaint narratives. Runs on Colab (GPU).

Self-contained on purpose: upload this file plus train/val/test.parquet and
split_meta.json to Colab, run it, download the `transformer_out/` folder, then
import the predictions into your local MLflow with:

    python scripts/phase2.py log-transformer --preds-dir path/to/transformer_out

Colab cells:
    !pip -q install transformers datasets accelerate pyarrow
    !python colab_train.py --data-dir . --out-dir transformer_out
    !zip -r transformer_out.zip transformer_out   # then download from the file pane

CPU smoke test (checks the script runs; not for real results):
    python scripts/colab_train.py --data-dir data/processed --out-dir /tmp/t --max-train 500 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

TEXT, TARGET = "narrative", "product"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True, help="folder with train/val/test.parquet + split_meta.json")
    p.add_argument("--out-dir", type=Path, default=Path("transformer_out"))
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--max-len", type=int, default=384, help="tokens; median narrative ~250 tokens")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-train", type=int, default=None, help="subsample train rows (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={a.model}")

    meta = json.loads((a.data_dir / "split_meta.json").read_text())
    classes: list[str] = meta["classes"]           # same order as the baseline
    label2id = {c: i for i, c in enumerate(classes)}

    splits = {s: pd.read_parquet(a.data_dir / f"{s}.parquet") for s in ("train", "val", "test")}
    if a.max_train:
        splits["train"] = splits["train"].sample(a.max_train, random_state=a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[[TEXT, TARGET]].reset_index(drop=True))
        ds = ds.map(lambda b: tok(b[TEXT], truncation=True, max_length=a.max_len), batched=True)
        ds = ds.map(lambda b: {"labels": [label2id[t] for t in b[TARGET]]}, batched=True)
        return ds.remove_columns([TEXT, TARGET])

    ds = {k: to_ds(v) for k, v in splits.items()}

    # Class weights (inverse frequency, normalised) to counter the 35:1 imbalance.
    counts = splits["train"][TARGET].value_counts()
    w = np.array([1.0 / counts.get(c, 1) for c in classes], dtype=np.float32)
    w = torch.tensor(w / w.mean())

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=label2id
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def macro_f1(eval_pred):
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        return {"macro_f1": f1_score(labels, logits.argmax(-1), average="macro")}

    steps_per_epoch = -(-len(ds['train']) // a.batch)
    warmup = int(0.06 * steps_per_epoch * a.epochs)

    args = TrainingArguments(
        output_dir=str(a.out_dir / "checkpoints"),
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        warmup_steps=warmup,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=a.seed,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok), compute_metrics=macro_f1,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    # Predictions for val and test, saved with complaint_id so they can be
    # aligned and scored locally by the shared evaluate() module.
    latency = {}
    for split in ("val", "test"):
        t0 = time.time()
        logits = trainer.predict(ds[split]).predictions
        latency[split] = (time.time() - t0) / len(ds[split]) * 1000
        proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        out = pd.DataFrame(proba, columns=classes)
        out.insert(0, "complaint_id", splits[split]["complaint_id"].to_numpy())
        out.to_parquet(a.out_dir / f"{split}_proba.parquet", index=False)

    trainer.save_model(a.out_dir / "model")
    tok.save_pretrained(a.out_dir / "model")
    (a.out_dir / "train_meta.json").write_text(json.dumps({
        "model": a.model, "max_len": a.max_len, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
        "n_train": len(ds["train"]), "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train_seconds": train_secs, "latency_ms_per_row": latency, "classes": classes,
    }, indent=2))
    print(f"done: train {train_secs:.0f}s, outputs in {a.out_dir}")


if __name__ == "__main__":
    main()
