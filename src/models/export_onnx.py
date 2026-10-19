"""Export a REAL trained checkpoint to ONNX and verify parity with PyTorch.

Refuses random-init (smoke) checkpoints unless --allow_random_init (tests only).
Parity: max |logit diff| and top-1 agreement on every image in --parity_dir
(default: the validation manifest). Writes <out>.parity.json. TFLite/INT8,
Android integration and device latency are NOT covered here.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from leaflens import config as cfg
from leaflens import inference as inf
from leaflens.checkpoint import load_checkpoint
from leaflens.data import read_manifest
from leaflens.manifest_paths import resolve_image_path


def export_and_check(ckpt, out, images, atol=1e-3):
    import onnxruntime as ort
    model, info = load_checkpoint(ckpt)
    x0 = torch.zeros(1, 3, inf.IMAGE_SIZE, inf.IMAGE_SIZE)
    torch.onnx.export(model, x0, str(out), input_names=["input"], output_names=["logits"], dynamo=True)  # fixed batch=1 (mobile use)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    diffs, agree = [], 0
    for p in images:
        x = inf.preprocess(p)
        with torch.inference_mode():
            want = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        diffs.append(float(np.abs(got - want).max()))
        agree += int(got.argmax() == want.argmax())
    rep = {"onnx": str(out), "checkpoint_sha256": info["sha256"], "n_images": len(diffs),
           "max_abs_logit_diff": max(diffs) if diffs else None,
           "top1_agreement": agree / len(diffs) if diffs else None, "atol": atol,
           "passed": bool(diffs) and max(diffs) <= atol and agree == len(diffs),
           "random_init_checkpoint": bool(info["meta"].get("random_init"))}
    Path(str(out) + ".parity.json").write_text(json.dumps(rep, indent=2))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{cfg.BASELINE_DIR}/{cfg.CHECKPOINT_NAME}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest_dir", default=cfg.MANIFEST_DIR)
    ap.add_argument("--root", default=".")
    ap.add_argument("--allow_random_init", action="store_true")
    a = ap.parse_args(argv)
    _, info = load_checkpoint(a.checkpoint)
    if info["meta"].get("random_init") and not a.allow_random_init:
        sys.exit("REFUSED: checkpoint is a random-init smoke artifact; export a real trained checkpoint")
    out = Path(a.out or Path(a.checkpoint).with_suffix(".onnx"))
    imgs = [resolve_image_path(r, a.root) for r in read_manifest(a.manifest_dir, "val").image_path]
    rep = export_and_check(a.checkpoint, out, imgs)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
