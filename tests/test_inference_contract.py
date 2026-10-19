import io
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from leaflens import inference as inf
from leaflens.manifest_paths import resolve_image_path
from data.preprocess import preprocess_image
from models import predict as predict_mod
from leaflens import checkpoint as ckmod


def _img(path, size=(300, 180), color=(30, 160, 40)):
    Image.new("RGB", size, color).save(path)
    return path


def test_pad_to_square_matches_training_preprocess(tmp_path):
    """Regression: old predict.py squashed (Resize((224,224))); training data
    was aspect-preserving pad-to-square. They must now be pixel-identical."""
    yy, xx = np.mgrid[0:180, 0:300]
    arr = np.stack([xx * 255 // 300, yy * 255 // 180, (xx + yy) % 256], -1).astype(np.uint8)
    src = tmp_path / "a.png"
    Image.fromarray(arr).save(src)
    out = tmp_path / "train_style.jpg"
    preprocess_image(src, out)  # training pipeline (saves JPEG q95)
    ours = inf.pad_to_square(inf.load_image(src))
    ref = Image.open(out).convert("RGB")
    diff = np.abs(np.asarray(ours, int) - np.asarray(ref, int))
    assert diff.mean() < 3.0  # only JPEG re-encoding noise remains
    assert ours.size == (224, 224)


def test_preprocess_shape_dtype_range(tmp_path):
    x = inf.preprocess(_img(tmp_path / "a.png"))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_exif_orientation_applied(tmp_path):
    im = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    p = tmp_path / "r.jpg"
    im.save(p, exif=exif)
    assert inf.load_image(p).size == (100, 200)


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\xff\xd8\xff garbage"])
def test_malformed_files_raise_invalid(tmp_path, payload):
    p = tmp_path / "bad.jpg"
    p.write_bytes(payload)
    with pytest.raises(inf.InvalidImageError):
        inf.load_image(p)


def test_tiny_image_rejected(tmp_path):
    with pytest.raises(inf.InvalidImageError):
        inf.load_image(_img(tmp_path / "t.png", size=(8, 8)))


def test_decide_single_temperature_single_softmax():
    logits = np.array([2.0, 1.0, 0.1])
    classes = {0: "a", 1: "b", 2: "c"}
    r = inf.decide(logits, classes, inf.Policy(threshold=0.5, temperature=2.0))
    z = logits / 2.0
    p = np.exp(z - z.max()); p /= p.sum()
    assert r["confidence"] == pytest.approx(round(float(p.max()), 4))
    assert r["prediction"] == "a"
    # applying softmax to probabilities (the double-softmax bug) differs
    p2 = np.exp(p) / np.exp(p).sum()
    assert abs(p2.max() - p.max()) > 0.01


def test_policy_statuses_and_validation():
    classes = {0: "a", 1: "b"}
    assert inf.decide([5, 0], classes, inf.Policy(0.7, 1.0))["status"] == "accepted"
    assert inf.decide([0.1, 0], classes, inf.Policy(0.7, 1.0))["status"] == "low_confidence"
    with pytest.raises(ValueError):
        inf.Policy(threshold=0)
    for bad_t in (-1, 0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            inf.Policy(threshold=0.5, temperature=bad_t)
    # no validated threshold => nothing is ever accepted
    assert inf.decide([50, 0], classes, inf.Policy(None, 1.0))["status"] == "low_confidence"


def test_decide_rejects_bad_shapes_and_nan():
    classes = {0: "a", 1: "b"}
    with pytest.raises(ValueError):
        inf.decide([1, 2, 3], classes, inf.Policy(0.5))
    with pytest.raises(ValueError):
        inf.decide([np.nan, 1], classes, inf.Policy(0.5))


def test_manifest_windows_paths():
    p = resolve_image_path("data\\preprocessed\\x\\train\\Apple Scab Leaf\\a.jpg", "/r")
    assert p.as_posix() == "/r/data/preprocessed/x/train/Apple Scab Leaf/a.jpg"
    assert resolve_image_path("a/b.jpg").as_posix() == "a/b.jpg"


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    classes = ["Class A", "Class B", "Class C"]
    m = ckmod.build_model(3, pretrained=False)
    p = tmp_path / "best_model.pth"
    ckmod.save_checkpoint(p, m, classes, {"random_init": True})
    return p


def test_end_to_end_deterministic_and_safe(tmp_path, ckpt):
    model, info = ckmod.load_checkpoint(ckpt)
    img = _img(tmp_path / "leaf.png")
    pol = inf.Policy(0.7, 1.267833, calibrated=True)
    r1 = predict_mod.predict(img, model, info, pol)
    r2 = predict_mod.predict(img, model, info, pol)
    assert r1 == r2
    assert r1["status"] in {"accepted", "low_confidence"}
    assert len(r1["top_predictions"]) == 3
    assert r1["random_init_checkpoint"] is True
    assert r1["image_quality_advisory"]["advisory_only"] is True
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"xx")
    assert predict_mod.predict(bad, model, info, pol)["status"] == "invalid_image"
    with pytest.raises(FileNotFoundError):
        predict_mod.predict(tmp_path / "nope.png", model, info, pol)


def test_quality_advisory_never_changes_decision(tmp_path, ckpt):
    model, info = ckmod.load_checkpoint(ckpt)
    img = _img(tmp_path / "flat.png")  # flat colour => "possibly blurry" advisory
    pol = inf.Policy(0.7, 1.0, calibrated=True)
    with_q = predict_mod.predict(img, model, info, pol, quality=True)
    no_q = predict_mod.predict(img, model, info, pol, quality=False)
    assert "possibly blurry" in with_q["image_quality_advisory"]["notes"]
    for k in ("status", "prediction", "confidence"):
        assert with_q[k] == no_q[k]


def test_onnx_parity_with_pytorch(tmp_path, ckpt):
    pytest.importorskip("onnxruntime")
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "src" / "models"))
    import export_onnx
    imgs = [_img(tmp_path / f"l{k}.png", size=(300 + 40 * k, 200), color=(30 + 50 * k, 160, 40))
            for k in range(3)]
    rep = export_onnx.export_and_check(ckpt, tmp_path / "m.onnx", imgs)
    assert rep["passed"] and rep["top1_agreement"] == 1.0
    assert rep["random_init_checkpoint"] is True


def test_export_refuses_random_init_checkpoint(tmp_path, ckpt):
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "src" / "models"))
    import export_onnx
    with pytest.raises(SystemExit, match="REFUSED"):
        export_onnx.main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "x.onnx")])
