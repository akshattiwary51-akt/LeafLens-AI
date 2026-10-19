"""Canonical training entry point (replaces train_mobilenetv3_v2/_v3 for new runs).

Three SEPARATE experiments (change one thing at a time):
  standard             plain shuffling, unweighted cross-entropy
  weighted_sampler     WeightedRandomSampler (inverse class frequency), unweighted CE
  class_weighted_loss  plain shuffling, inverse-frequency-weighted CE

Model selection metric (predeclared): validation macro-F1. The test split is
never read here. Requires ImageNet-pretrained weights; without them the run
STOPS (exit 2) unless --allow_random_init is passed, which stamps the
checkpoint ``random_init=True`` for pipeline smoke tests only.

  python src/models/train.py --experiment standard --output_dir reports/mobilenetv3_baseline
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from leaflens import config as cfg
from leaflens.checkpoint import build_model, save_checkpoint
from leaflens.data import ManifestDataset, load_classes, read_manifest


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels, n):
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    w = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in loader:
            ps.append(model(x).argmax(1).numpy())
            ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean()), float(f1_score(y, p, average="macro"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pretrained_available():
    try:
        build_model(2, pretrained=True)
        return True, ""
    except Exception as e:  # network / cache failure
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    d = cfg.TrainConfig()
    ap.add_argument("--manifest_dir", default=d.manifest_dir)
    ap.add_argument("--root", default=d.root)
    ap.add_argument("--experiment", choices=cfg.EXPERIMENTS, default=d.experiment)
    ap.add_argument("--output_dir", default=d.output_dir)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch_size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--weight_decay", type=float, default=d.weight_decay)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--num_workers", type=int, default=d.num_workers)
    ap.add_argument("--allow_random_init", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.output_dir)
    if (out / cfg.CHECKPOINT_NAME).exists():
        sys.exit(f"refusing to overwrite existing checkpoint in {out}; choose another --output_dir")

    ok, why = pretrained_available()
    if not ok and not a.allow_random_init:
        print("BLOCKED: ImageNet-pretrained MobileNetV3-Small weights are unavailable.\n"
              f"  reason: {why}\n  next: run once with network access to download.pytorch.org, or place\n"
              "  mobilenet_v3_small-047dcff4.pth in ~/.cache/torch/hub/checkpoints/ and rerun.",
              file=sys.stderr)
        sys.exit(2)
    pretrained = ok

    set_seed(a.seed)
    classes, c2i = load_classes(a.manifest_dir)
    tr_df, va_df = (read_manifest(a.manifest_dir, s) for s in ("train", "val"))
    tr = ManifestDataset(tr_df, c2i, a.root, train=True)
    va = ManifestDataset(va_df, c2i, a.root, train=False)
    g = torch.Generator().manual_seed(a.seed)
    if a.experiment == "weighted_sampler":
        w = class_weights(tr.labels, len(classes))[torch.from_numpy(tr.labels)]
        sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True, generator=g)
        tl = DataLoader(tr, a.batch_size, sampler=sampler, num_workers=a.num_workers)
    else:
        tl = DataLoader(tr, a.batch_size, shuffle=True, generator=g, num_workers=a.num_workers)
    vl = DataLoader(va, 64, num_workers=a.num_workers)

    model = build_model(len(classes), pretrained)
    loss_w = class_weights(tr.labels, len(classes)) if a.experiment == "class_weighted_loss" else None
    crit = nn.CrossEntropyLoss(weight=loss_w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
        sched.step()
        acc, f1 = evaluate(model, vl)
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), val_acc=acc, val_macro_f1=f1))
        print(json.dumps(hist[-1]), flush=True)
        if f1 > best:
            best, best_ep, bad, best_state = f1, ep, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    meta = dict(experiment=a.experiment, seed=a.seed, selection_metric=cfg.SELECTION_METRIC,
                best_val_macro_f1=best, best_epoch=best_ep, epochs_run=len(hist),
                train_seconds=round(time.time() - t0, 1), pretrained=pretrained,
                random_init=not pretrained,
                train_manifest_sha256=sha(Path(a.manifest_dir) / "train_manifest.csv"),
                val_manifest_sha256=sha(Path(a.manifest_dir) / "val_manifest.csv"),
                hyperparameters=vars(a))
    save_checkpoint(out / cfg.CHECKPOINT_NAME, model, classes, meta)
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out/cfg.CHECKPOINT_NAME} (best val macro-F1 {best:.4f} @ epoch {best_ep}"
          f"{'; RANDOM INIT SMOKE RUN' if not pretrained else ''})")


if __name__ == "__main__":
    main()
