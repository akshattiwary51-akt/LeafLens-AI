"""Real unit test for src/data/audit.py phash function."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.audit import phash
from PIL import Image, ImageDraw


def _textured_image(seed_val):
    """A solid-color image has zero variance, so every pixel sits exactly
    on the mean and average-hash degenerates to all-zero-bits regardless
    of the color — a real edge case this test caught. Use a textured
    image (closer to an actual leaf photo) instead."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(0, 100, 10):
        draw.line([(i + seed_val, 0), (i + seed_val, 100)], fill=(0, 0, 0), width=3)
    return img


def test_phash_identical_images_match():
    img1 = _textured_image(0)
    img2 = _textured_image(0)
    assert phash(img1) == phash(img2)


def test_phash_different_images_differ():
    img1 = _textured_image(0)
    img2 = _textured_image(5)
    assert phash(img1) != phash(img2)


def test_phash_solid_color_is_a_known_degenerate_case():
    """Documents the actual finding above rather than hiding it: a
    zero-variance (solid-color) image hashes to all-zero bits no matter
    the color, so two different solid colors collide. Real photos are
    never perfectly flat, so this hasn't caused false duplicate matches
    in the PlantDoc audit, but it's a documented limitation of aHash."""
    white = Image.new("RGB", (100, 100), color=(255, 255, 255))
    black = Image.new("RGB", (100, 100), color=(0, 0, 0))
    assert phash(white) == phash(black)  # documents the degenerate case


if __name__ == "__main__":
    test_phash_identical_images_match()
    test_phash_different_images_differ()
    test_phash_solid_color_is_a_known_degenerate_case()
    print("All audit tests passed.")
