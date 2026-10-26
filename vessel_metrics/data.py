import csv
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".gif", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".ppm"}


def list_images(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return {p.stem: p for p in files}


def list_subfolders(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_dir())


def read_mask(path):
    """Read a binary mask. Pixels above half of the maximum value are considered foreground."""

    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    if img.max() == 0:
        return img.astype(bool)
    return img > img.max() / 2


def read_probability(path):
    """Read a probability map stored as the first array of a .npz file."""

    with np.load(path) as data:
        prob = data[list(data.keys())[0]]
    prob = np.squeeze(prob).astype(np.float32)
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError(f"Values of {path} are outside the range [0, 1].")

    return prob


class CsvWriter:
    def __init__(self, path, fieldnames):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
