import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

import build_manifests as bm
import duplicate_groups as dg
import preprocess as pp
import verify_manifests as vm
from dataset_audit import dhash

ROOT = Path(__file__).resolve().parents[1]


def _photo(seed, size=(120, 80)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.Resampling.BICUBIC)


def _make_raw(tmp):
    """train/ and test/ with 2 classes, duplicates, collisions, unsafe names."""
    raw = tmp / "raw"
    for split, cls, name, seed in [
        ("train", "A", "a1.jpg", 1), ("train", "A", "a2.jpg", 2), ("train", "A", "a3.jpg", 3),
        ("train", "A", "a4.jpg", 4), ("train", "A", "a5.jpg", 5), ("train", "A", "a6.jpg", 6),
        ("train", "B", "b1.jpg", 11), ("train", "B", "b2.jpg", 12), ("train", "B", "b3.jpg", 13),
        ("train", "B", "b4.jpg", 14), ("train", "B", "b5.jpg", 15), ("train", "B", "b6.jpg", 16),
        ("train", "B", "dup_of_test.jpg", 21),   # same photo as the test image below
        ("train", "B", "conf_b.jpg", 31),        # same photo filed under A and B
        ("train", "A", "conf_a.jpg", 31),
        ("test", "A", "t_a.jpg", 41), ("test", "A", "t_a2.jpg", 42),
        ("test", "B", "t_b.jpg", 21),
        ("test", "B", "t_b2.jpg", 43),
    ]:
        p = raw / split / cls / name
        p.parent.mkdir(parents=True, exist_ok=True)
        _photo(seed).save(p, quality=95)
    return raw


def _pipeline(tmp):
    raw = _make_raw(tmp)
    out = tmp / "pre"
    rows = []
    for s in ("train", "test"):
        r, f = pp.process_split(raw / s, out, s, 224, raw)
        assert not f
        rows += r
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.to_csv(out / "preprocess_index.csv", index=False)
    hashes = []
    for o in df.output:
        with Image.open(out / o) as im:
            hashes.append(dhash(im.convert("RGB")))
    pairs = dg.compute_pairs(df, hashes, 4)
    groups, _ = dg.build_groups(df, pairs)
    return raw, out, df, groups


def test_safe_stem_and_collision_free_names(tmp_path):
    raw = tmp_path / "raw"
    for n in ["CAR1.jpg", "car1.jpg", "x.png", "x.jpg", 'we?ird:name*.jpeg']:
        p = raw / "A" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        _photo(1).save(p, format="PNG" if n.endswith("png") else "JPEG")
    imgs = pp.collect_images(raw)
    plan = pp.plan_output_names(imgs, raw)
    names = [v[0] for v in plan.values()]
    assert len(set(n.lower() for n in names)) == len(names)  # nothing overwritten
    assert sum(v[1] for v in plan.values()) == 4  # CAR1/car1 and x.png/x.jpg
    weird = [v[0] for k, v in plan.items() if "ird" in k.name][0]
    assert not any(c in weird for c in '?:*<>|"')
    # deterministic regardless of input order
    assert pp.plan_output_names(list(reversed(imgs)), raw) == plan


def test_preprocess_applies_exif_orientation(tmp_path):
    im = Image.new("RGB", (200, 100), (255, 0, 0))
    ex = Image.Exif(); ex[0x0112] = 6
    src = tmp_path / "e.jpg"; im.save(src, exif=ex)
    out = tmp_path / "o.jpg"
    assert pp.preprocess_image(src, out) == (100, 200)  # rotated source size
    arr = np.asarray(Image.open(out).convert("RGB"))
    # rotated image is tall: red columns in the middle, black padding left/right
    assert arr[112, 5].sum() < 40 and arr[112, 112, 0] > 200


def test_groups_cover_exact_dups_and_label_conflicts(tmp_path):
    _, _, df, groups = _pipeline(tmp_path)
    g = groups.set_index("output")
    k = lambda s, c, n: f"{s}/{c}/{n}.jpg"
    # train duplicate of a test image shares a group and crosses splits
    assert g.loc[k("train", "B", "dup_of_test"), "group_id"] == g.loc[k("test", "B", "t_b"), "group_id"]
    assert g.loc[k("test", "B", "t_b"), "crosses_splits"]
    # same photo under two classes -> label conflict
    assert g.loc[k("train", "A", "conf_a"), "label_conflict"]


def test_build_manifests_no_leakage_and_records_quarantine(tmp_path):
    raw, out, df, groups = _pipeline(tmp_path)
    g, classes = bm.build(groups, min_class_images=3, val_folds=3, seed=1)
    mdir = tmp_path / "man"; mdir.mkdir()
    rep = bm.write(g, classes, mdir, "pre")
    assert rep["quarantined"]["label_conflict_pending_review"] == 2
    assert rep["quarantined"]["leaks_into_frozen_test"] == 1
    q = pd.read_csv(mdir / "quarantine.csv")
    assert len(q) == 3  # nothing disappears without a record
    assert vm.verify(mdir, out.parent) == []
    # frozen test cannot be silently changed
    t = pd.read_csv(mdir / "test_manifest.csv")
    t.iloc[:-1].to_csv(mdir / "test_manifest.csv", index=False)
    assert any("freezing" in p for p in vm.verify(mdir, out.parent))


def test_verify_detects_missing_file_and_group_crossing(tmp_path):
    raw, out, df, groups = _pipeline(tmp_path)
    g, classes = bm.build(groups, min_class_images=3, val_folds=3, seed=1)
    mdir = tmp_path / "man"; mdir.mkdir()
    bm.write(g, classes, mdir, "pre")
    victim = pd.read_csv(mdir / "train_manifest.csv").image_path[0]
    (tmp_path / victim).unlink()
    assert any("missing file" in p for p in vm.verify(mdir, tmp_path))
    # force a leaked group into val
    tr = pd.read_csv(mdir / "train_manifest.csv")
    va = pd.read_csv(mdir / "val_manifest.csv")
    leak = tr.iloc[[1]].assign(split="val")
    pd.concat([va, leak]).to_csv(mdir / "val_manifest.csv", index=False)
    assert any("group leakage" in p for p in vm.verify(mdir, tmp_path))


def test_label_mapping_is_dense_sorted(tmp_path):
    _, out, df, groups = _pipeline(tmp_path)
    g, classes = bm.build(groups, min_class_images=3, val_folds=3, seed=1)
    mdir = tmp_path / "man"; mdir.mkdir()
    bm.write(g, classes, mdir, "pre")
    m = json.loads((mdir / "classes.json").read_text())
    assert m["classes"] == sorted(m["classes"])
    assert sorted(m["class_to_idx"].values()) == list(range(len(m["classes"])))


def test_review_decision_can_split_a_false_positive_group(tmp_path):
    _, out, df, groups = _pipeline(tmp_path)
    a, b = "train/A/conf_a.jpg", "train/B/conf_b.jpg"
    dec = pd.DataFrame([dict(a=a, b=b, decision="different")])
    pairs = dg.compute_pairs(df, [dhash(Image.open(out / o).convert("RGB")) for o in df.output], 4)
    g2, _ = dg.build_groups(df, pairs, dec)
    assert not g2.set_index("output").loc[a, "label_conflict"]


@pytest.mark.skipif(not (ROOT / "data/preprocessed/plantdoc_224_v2").exists(),
                    reason="preprocessed dataset not present")
def test_active_v2_manifests_verify_clean():
    assert vm.verify(ROOT / "data/manifests_v2", ROOT) == []
