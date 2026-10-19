"""Datasets for the canonical pipeline (v2 manifests)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import inference as inf
from .manifest_paths import resolve_image_path


def load_classes(manifest_dir):
    m = json.loads((Path(manifest_dir) / "classes.json").read_text())
    return m["classes"], m["class_to_idx"]


def read_manifest(manifest_dir, split):
    return pd.read_csv(Path(manifest_dir) / f"{split}_manifest.csv")


class ManifestDataset(Dataset):
    """train=True: augmentation on the padded 224 image, then the SAME
    ToTensor+Normalize numbers as inference. train=False: exactly the
    inference path (leaflens.inference.pad_to_square + to_array)."""

    def __init__(self, df, class_to_idx, root=".", train=False):
        self.df = df.reset_index(drop=True)
        self.c2i = class_to_idx
        self.root = root
        self.train = train
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(inf.IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        ])

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return np.array([self.c2i[c] for c in self.df["class"]])

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = inf.pad_to_square(inf.load_image(resolve_image_path(r.image_path, self.root)))
        if self.train:
            im = self.aug(im)
        x = torch.from_numpy(inf.to_array(im)[0])
        return x, self.c2i[r["class"]]
