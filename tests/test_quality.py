"""Real unit tests for src/preprocessing/quality.py, run against actual
downloaded PlantDoc images (not synthetic fixtures) where available."""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.quality import assess_quality


def test_assess_quality_on_real_images():
    candidates = glob.glob("data/raw/plantdoc/test/*/*.jpg")[:5]
    if not candidates:
        print("SKIP: no downloaded PlantDoc images found — run src/data/download.py plantdoc first")
        return
    for path in candidates:
        result = assess_quality(path)
        assert result["quality"] in {"GOOD", "FAIR", "POOR", "UNREADABLE"}
        assert "width" in result and "height" in result
        print(path, "->", result["quality"], result["reasons"])


def test_missing_file_handled():
    result = assess_quality("data/raw/plantdoc/does_not_exist.jpg")
    assert result["quality"] == "UNREADABLE"


if __name__ == "__main__":
    test_assess_quality_on_real_images()
    test_missing_file_handled()
    print("All quality tests passed.")
