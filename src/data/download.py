"""
Dataset download helper.

In this research sandbox, only github.com/codeload.github.com/
raw.githubusercontent.com (plus package registries) were reachable —
PlantDoc was downloaded this way (see reports/dataset_research.md).
PlantVillage (Hugging Face) and Paddy Doctor (Kaggle) need to be pulled
from the real dev environment, which will have unrestricted network
access. This script documents exactly how, so Milestone 2 doesn't have
to re-derive it.
"""
import argparse
import subprocess


SOURCES = {
    "plantdoc": {
        "method": "git",
        "cmd": ["git", "clone", "--depth", "1",
                "https://github.com/pratikkayal/PlantDoc-Dataset.git"],
        "note": "CC BY 4.0. Verified working from this sandbox.",
    },
    "plantvillage": {
        "method": "huggingface",
        "cmd": None,
        "note": (
            "Not reachable from this sandbox (huggingface.co not on the network "
            "allowlist). In the real dev environment: "
            "`pip install datasets` then "
            "`from datasets import load_dataset; "
            "ds = load_dataset('mohanty/PlantVillage', 'color')`. "
            "~54,300 images, ~15-20GB estimated — budget time/disk accordingly."
        ),
    },
    "paddy_doctor": {
        "method": "kaggle",
        "cmd": None,
        "note": (
            "Not reachable from this sandbox (Kaggle not on the network allowlist). "
            "In the real dev environment: `kaggle competitions download -c "
            "paddy-disease-classification` (requires a Kaggle API token). "
            "16,225 images per paddydoc.github.io."
        ),
    },
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SOURCES.keys()))
    ap.add_argument("--dest", default="data/raw")
    args = ap.parse_args()

    entry = SOURCES[args.dataset]
    print(f"[{args.dataset}] {entry['note']}")
    if entry["cmd"]:
        subprocess.run(entry["cmd"], cwd=args.dest, check=True)
    else:
        print("No automated download available from this environment — see note above.")
