"""Calculate the metrics between the predictions of segmentation models and the ground truth."""

import argparse
import traceback

from vessel_metrics.compute import METRIC_NAMES, compute_metrics
from vessel_metrics.data import CsvWriter, list_images, list_subfolders, read_mask, read_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/labels", help="Folder with one subfolder of masks per dataset.")
    parser.add_argument("--predictions", default="data/predictions",
                        help="Folder with one subfolder per dataset, each containing one subfolder of .npz files per model.")
    parser.add_argument("--output", default="results/models.csv")
    parser.add_argument("--datasets", nargs="*", help="Datasets to process. Default: all subfolders.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    fieldnames = ["dataset", "model", "image"] + METRIC_NAMES
    writer = CsvWriter(args.output, fieldnames)

    for dataset_dir in list_subfolders(args.predictions):
        if args.datasets and dataset_dir.name not in args.datasets:
            continue
        labels = list_images(f"{args.labels}/{dataset_dir.name}")
        for model_dir in list_subfolders(dataset_dir):
            for pred_path in sorted(model_dir.glob("*.npz")):
                name = pred_path.stem
                if name not in labels:
                    print(f"Missing label for {pred_path}", flush=True)
                    continue
                print(f"{dataset_dir.name} - {model_dir.name} - {name}", flush=True)
                try:
                    prob = read_probability(pred_path)
                    target = read_mask(labels[name])
                    row = {"dataset": dataset_dir.name, "model": model_dir.name, "image": name}
                    row.update(compute_metrics(prob > args.threshold, target, prob))
                    writer.write(row)
                except Exception:
                    print(f"Failed on {pred_path}:\n{traceback.format_exc()}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()
