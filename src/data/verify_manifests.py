"""Verify active manifests: paths, decode, labels, group/hash leakage, frozen test.

Exit code 1 if any problem is found. Usage:
  python src/data/verify_manifests.py [--manifests data/manifests_v2] [--root .]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from leaflens.manifest_paths import resolve_image_path


def verify(manifest_dir, root="."):
    md = Path(manifest_dir)
    problems = []
    classes = json.loads((md / "classes.json").read_text())["classes"]
    frames = {s: pd.read_csv(md / f"{s}_manifest.csv") for s in ("train", "val", "test")}
    for s, df in frames.items():
        if df.image_path.duplicated().any():
            problems.append(f"{s}: duplicate image_path rows")
        bad = set(df["class"]) - set(classes)
        if bad:
            problems.append(f"{s}: unknown classes {sorted(bad)}")
        if (df.split != s).any():
            problems.append(f"{s}: split column mismatch")
        for r in df.itertuples():
            p = resolve_image_path(r.image_path, root)
            if not p.exists():
                problems.append(f"{s}: missing file {r.image_path}")
                continue
            try:
                with Image.open(p) as im:
                    im.load()
                    if im.size != (224, 224) or im.mode != "RGB":
                        problems.append(f"{s}: unexpected size/mode {im.size} {im.mode} {r.image_path}")
            except Exception as e:
                problems.append(f"{s}: undecodable {r.image_path}: {e}")
        for c in classes:
            if (df["class"] == c).sum() == 0:
                problems.append(f"{s}: class '{c}' has no images")
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga, gb = set(frames[a].group_id), set(frames[b].group_id)
            if ga & gb:
                problems.append(f"group leakage {a}/{b}: {len(ga & gb)} groups")
            ha, hb = set(frames[a].source_sha256), set(frames[b].source_sha256)
            if ha & hb:
                problems.append(f"exact-duplicate leakage {a}/{b}: {len(ha & hb)} hashes")
    allf = pd.concat(frames.values())
    conflict = allf.groupby("group_id")["class"].nunique()
    if (conflict > 1).any():
        problems.append("label conflicts inside active manifests")
    frozen = md / "TEST_FROZEN.sha256"
    if not frozen.exists():
        problems.append("TEST_FROZEN.sha256 missing")
    elif frozen.read_text().strip() != hashlib.sha256((md / "test_manifest.csv").read_bytes()).hexdigest():
        problems.append("test manifest changed after freezing")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests_v2")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    p = verify(a.manifests, a.root)
    for x in p[:50]:
        print("FAIL:", x)
    print("manifests OK" if not p else f"{len(p)} problem(s)")
    sys.exit(1 if p else 0)


if __name__ == "__main__":
    main()
