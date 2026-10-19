"""Preprocess raw PlantDoc images to the canonical 224x224 padded format.

Pipeline per image (shared implementation: src/leaflens/inference.py):
  decode -> EXIF orientation -> RGB -> aspect-preserving resize -> centre on a
  black square canvas -> JPEG q95.

Collision/portability safety (see tests/test_preprocess.py):
  * output names are sanitised for Windows/macOS/Linux ("?", ":", ... -> "_")
  * two sources that map to the same output name (case-insensitive, per class)
    BOTH get a deterministic ``__<hash8>`` suffix -- nothing is overwritten
  * ``preprocess_index.csv`` maps every source file to its output file
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from leaflens.inference import IMAGE_SIZE, load_image, pad_to_square

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def collect_images(data_dir):
    """Return [(path, class_name)] for supported images, sorted."""
    data_dir = Path(data_dir)
    out = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                out.append((p, class_dir.name))
    return out


def safe_stem(name):
    s = _UNSAFE.sub("_", Path(name).stem).rstrip(" .")
    return (s or "image")[:150]


def plan_output_names(images, root):
    """Deterministic, collision-free output file names.

    Returns {source_path: (output_name, collided)}.
    """
    groups = {}
    for p, cls in images:
        groups.setdefault((cls, safe_stem(p.name).lower()), []).append(p)
    plan = {}
    for (cls, _), members in groups.items():
        for p in members:
            stem = safe_stem(p.name)
            collided = len(members) > 1
            if collided:
                rel = p.relative_to(root).as_posix()
                stem += "__" + hashlib.sha1(rel.encode()).hexdigest()[:8]
            plan[p] = (stem + ".jpg", collided)
    return plan


def preprocess_image(image_path, output_path, image_size=IMAGE_SIZE):
    """Decode (EXIF-aware), pad to square, save JPEG. Returns source size."""
    im = load_image(image_path)
    src_size = im.size
    out = pad_to_square(im, image_size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, format="JPEG", quality=95)
    return src_size


def process_split(input_dir, output_dir, split, image_size, root):
    images = collect_images(input_dir)
    plan = plan_output_names(images, Path(root))
    rows, failed = [], []
    for p, cls in images:
        name, collided = plan[p]
        out = Path(output_dir) / split / cls / name
        try:
            w, h = preprocess_image(p, out, image_size)
            rows.append(dict(split=split, cls=cls,
                             source=p.relative_to(root).as_posix(),
                             output=out.relative_to(output_dir).as_posix(),
                             source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             src_w=w, src_h=h, name_collision=collided))
        except Exception as e:  # recorded, never silently skipped
            failed.append(dict(split=split, cls=cls, source=str(p), error=str(e)))
    return rows, failed


def main():
    ap = argparse.ArgumentParser(description="Preprocess PlantDoc dataset")
    ap.add_argument("--input_dir", default="data/raw/PlantDoc-Dataset",
                    help="folder containing train/ and test/")
    ap.add_argument("--output_dir", default="data/preprocessed/plantdoc_224_v2")
    ap.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    a = ap.parse_args()
    root = Path(a.input_dir)
    out = Path(a.output_dir)
    rows, failed = [], []
    for split in ("train", "test"):
        r, f = process_split(root / split, out, split, a.image_size, root)
        rows += r
        failed += f
        print(f"{split}: processed={len(r)} failed={len(f)}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "preprocess_index.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(out / "preprocess_failures.csv", index=False)
    print(f"collisions resolved: {sum(r['name_collision'] for r in rows)}; index: {out/'preprocess_index.csv'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
