"""Calculate the metrics between ground truth masks and their perturbed versions."""

import argparse
import traceback

import numpy as np

from vessel_metrics.compute import METRIC_NAMES, compute_metrics
from vessel_metrics.data import CsvWriter, list_images, list_subfolders, read_mask
from vessel_metrics.perturbations import border_perturbation, create_noisy_img, random_crop

NOISE_LEVELS = np.round(np.arange(0, 1, 0.1), 1)


def perturbations(img, rng, max_dist):
    for p in NOISE_LEVELS:
        yield "spike", p, create_noisy_img(img, p, 0, 0, rng)
    for p in NOISE_LEVELS:
        yield "fp", p, create_noisy_img(img, 0, p, 0, rng)
    for p in NOISE_LEVELS:
        yield "fn", p, create_noisy_img(img, 0, 0, p, rng)
    for mode in ["erode", "dilate"]:
        yield mode, 0., img
        imgs, dists = border_perturbation(img, max_dist, mode)
        for img_pert, dist in zip(imgs, dists):
            yield mode, dist, img_pert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/labels", help="Folder with one subfolder of masks per dataset.")
    parser.add_argument("--output", default="results/noise.csv")
    parser.add_argument("--datasets", nargs="*", help="Datasets to process. Default: all subfolders.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--min-vessel-fraction", type=float, default=0.1)
    parser.add_argument("--max-dist", type=float, default=9.)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fieldnames = ["dataset", "image", "perturbation", "level", "area_change"] + METRIC_NAMES
    writer = CsvWriter(args.output, fieldnames)

    for dataset_dir in list_subfolders(args.labels):
        if args.datasets and dataset_dir.name not in args.datasets:
            continue
        for name, path in list_images(dataset_dir).items():
            print(f"{dataset_dir.name} - {name}", flush=True)
            try:
                img = random_crop(read_mask(path), rng, args.crop_size, args.min_vessel_fraction)
                for perturbation, level, img_pert in perturbations(img, rng, args.max_dist):
                    row = {
                        "dataset": dataset_dir.name,
                        "image": name,
                        "perturbation": perturbation,
                        "level": float(level),
                        "area_change": (img_pert.sum() - img.sum()) / img.size,
                    }
                    row.update(compute_metrics(img_pert, img))
                    writer.write(row)
            except Exception:
                print(f"Failed on {path}:\n{traceback.format_exc()}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()
