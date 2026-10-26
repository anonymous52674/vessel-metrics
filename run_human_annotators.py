"""Calculate the metrics between the annotations of two human annotators. Each metric is
calculated using each annotator as the reference and the two values are averaged."""

import argparse
import traceback

from vessel_metrics.compute import METRIC_NAMES, compute_metrics
from vessel_metrics.data import CsvWriter, list_images, list_subfolders, read_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotators",
                        help="Folder with one subfolder per dataset, each containing the subfolders annotator1 and annotator2.")
    parser.add_argument("--output", default="results/human_annotators.csv")
    parser.add_argument("--datasets", nargs="*", help="Datasets to process. Default: all subfolders.")
    args = parser.parse_args()

    fieldnames = ["dataset", "image", "area_change"] + METRIC_NAMES
    writer = CsvWriter(args.output, fieldnames)

    for dataset_dir in list_subfolders(args.annotations):
        if args.datasets and dataset_dir.name not in args.datasets:
            continue
        images1 = list_images(dataset_dir / "annotator1")
        images2 = list_images(dataset_dir / "annotator2")
        for name, path1 in images1.items():
            if name not in images2:
                print(f"Missing annotator2 image for {path1}", flush=True)
                continue
            print(f"{dataset_dir.name} - {name}", flush=True)
            try:
                img1 = read_mask(path1)
                img2 = read_mask(images2[name])
                metrics_1 = compute_metrics(img2, img1)
                metrics_2 = compute_metrics(img1, img2)
                row = {
                    "dataset": dataset_dir.name,
                    "image": name,
                    "area_change": abs(int(img2.sum()) - int(img1.sum())) / img1.size,
                }
                row.update({k: (metrics_1[k] + metrics_2[k]) / 2 for k in METRIC_NAMES})
                writer.write(row)
            except Exception:
                print(f"Failed on {path1}:\n{traceback.format_exc()}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()
