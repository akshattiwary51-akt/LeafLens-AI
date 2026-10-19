import json
import sys
from pathlib import Path

import pytest
import torch

from leaflens import checkpoint as ck
from leaflens import config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
import train as train_mod  # noqa: E402


def _tiny_ckpt(path, classes=("a", "b", "c")):
    torch.manual_seed(0)
    m = ck.build_model(len(classes), pretrained=False)
    ck.save_checkpoint(path, m, list(classes), {"random_init": True})
    return m


def test_roundtrip_and_hash(tmp_path):
    p = tmp_path / "m.pth"
    m = _tiny_ckpt(p)
    m2, info = ck.load_checkpoint(p)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        assert torch.equal(m.eval()(x), m2(x))
    assert info["classes"] == ["a", "b", "c"] and len(info["sha256"]) == 64


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.update(schema_version=99), "schema"),
    (lambda d: d.pop("classes"), "missing"),
    (lambda d: d.update(class_to_idx={"a": 0, "b": 2, "c": 1}), "disagrees"),
    (lambda d: d.update(class_to_idx={"a": 0, "b": 0, "c": 1}), "dense"),
    (lambda d: d.update(preprocessing_id="squash"), "preprocessing"),
    (lambda d: d.update(arch="resnet"), "arch"),
    (lambda d: d.update(classes=["a", "a", "c"]), "duplicates"),
])
def test_malformed_checkpoints_rejected(tmp_path, mutate, msg):
    p = tmp_path / "m.pth"
    _tiny_ckpt(p)
    d = torch.load(p, weights_only=True)
    mutate(d)
    torch.save(d, p)
    with pytest.raises(ck.CheckpointError, match=msg):
        ck.load_checkpoint(p)


def test_wrong_class_count_and_garbage_and_missing(tmp_path):
    p = tmp_path / "m.pth"
    _tiny_ckpt(p)
    d = torch.load(p, weights_only=True)
    d["classes"] = ["a", "b"]; d["class_to_idx"] = {"a": 0, "b": 1}; d["num_classes"] = 2
    torch.save(d, p)
    with pytest.raises(ck.CheckpointError, match="do not fit"):
        ck.load_checkpoint(p)
    (tmp_path / "junk.pth").write_bytes(b"not a checkpoint")
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(tmp_path / "junk.pth")
    with pytest.raises(FileNotFoundError):
        ck.load_checkpoint(tmp_path / "nope.pth")


def test_legacy_pickle_checkpoint_not_loaded_unsafely(tmp_path):
    class Evil:
        def __reduce__(self):
            return (exec, ("raise SystemExit('pwned')",))
    p = tmp_path / "evil.pth"
    torch.save({"x": Evil()}, p)
    with pytest.raises(ck.CheckpointError):  # weights_only refuses arbitrary objects
        ck.load_checkpoint(p)


def test_class_weights_inverse_frequency():
    import numpy as np
    w = train_mod.class_weights(np.array([0] * 8 + [1] * 2), 2)
    assert w[1] / w[0] == pytest.approx(4.0)


def test_training_blocks_without_pretrained_weights(tmp_path, tiny_dataset, monkeypatch, capsys):
    monkeypatch.setattr(train_mod, "pretrained_available", lambda: (False, "offline"))
    with pytest.raises(SystemExit) as e:
        train_mod.main(["--manifest_dir", str(tiny_dataset["manifests"]), "--root",
                        str(tiny_dataset["root"]), "--output_dir", str(tmp_path / "o")])
    assert e.value.code == 2
    assert "BLOCKED" in capsys.readouterr().err
    assert not (tmp_path / "o" / cfg.CHECKPOINT_NAME).exists()


@pytest.mark.parametrize("exp", cfg.EXPERIMENTS)
def test_smoke_training_each_experiment_is_stamped_and_reproducible(tmp_path, tiny_dataset, monkeypatch, exp):
    monkeypatch.setattr(train_mod, "pretrained_available", lambda: (False, "offline"))
    monkeypatch.setattr(train_mod, "build_model",
                        lambda n, pretrained: ck.build_model(n, False))
    outs = []
    for k in range(2):
        o = tmp_path / f"o{k}"
        train_mod.main(["--manifest_dir", str(tiny_dataset["manifests"]), "--root",
                        str(tiny_dataset["root"]), "--output_dir", str(o), "--experiment", exp,
                        "--epochs", "1", "--batch_size", "8", "--allow_random_init", "--seed", "7"])
        outs.append(o)
    m = json.loads((outs[0] / "train_meta.json").read_text())
    assert m["random_init"] is True and m["experiment"] == exp
    assert m["selection_metric"] == "val_macro_f1"
    a = torch.load(outs[0] / cfg.CHECKPOINT_NAME, weights_only=True)["model_state_dict"]
    b = torch.load(outs[1] / cfg.CHECKPOINT_NAME, weights_only=True)["model_state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)  # same seed => same weights
    with pytest.raises(SystemExit):  # never overwrites
        train_mod.main(["--manifest_dir", str(tiny_dataset["manifests"]), "--root",
                        str(tiny_dataset["root"]), "--output_dir", str(outs[0]),
                        "--allow_random_init"])
