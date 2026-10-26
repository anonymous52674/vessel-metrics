import numpy as np

from .confusion import average_precision, confusion_matrix_metrics, matthews_corrcoef, roc_auc
from .distance import distance_metrics
from .hce import hce
from .topology import apls, cl_dice, wiedemann_metrics

METRIC_NAMES = [
    "ACC", "PRE", "REC", "DIC", "IOU", "RAC", "APR", "MCC", "HCE", "HAU", "DGP", "DPG",
    "CDC", "COM", "COR", "QUA", "OFG", "OFP", "SDC", "APL",
]


def compute_metrics(
        pred,
        target,
        prob=None,
        buffer_width=3.,
        surface_tolerance=0.,
        hausdorff_percent=100.,
        apls_nodes=12,
        ):
    """Calculate all metrics between a predicted and a target binary mask.

    Parameters
    ----------
    pred
        Binary predicted mask.
    target
        Binary target mask.
    prob
        Probability map predicted by the model. If None, RAC and APR are calculated from the
        binary prediction, which corresponds to a single threshold of 0.5.
    buffer_width
        Buffer width used in the completeness, correctness and quality metrics.
    surface_tolerance
        Tolerance used in the surface overlap and surface Dice metrics.
    hausdorff_percent
        Percentile of the distances used in the Hausdorff distance.
    apls_nodes
        Maximum number of ground truth nodes used in the APLS metric.

    Returns
    -------
        A dictionary with the value of each metric in METRIC_NAMES.
    """

    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} differs from target shape {target.shape}.")

    if prob is None:
        prob = pred.astype(np.float32)

    results = confusion_matrix_metrics(pred, target)
    results["RAC"] = roc_auc(prob, target)
    results["APR"] = average_precision(prob, target)
    results["MCC"] = matthews_corrcoef(pred, target)
    results["HCE"] = hce(pred, target)
    results.update(distance_metrics(pred, target, surface_tolerance, hausdorff_percent))
    results["CDC"] = cl_dice(pred, target)
    results.update(wiedemann_metrics(pred, target, buffer_width))
    results["APL"] = apls(pred, target, apls_nodes)

    return {name: float(results[name]) for name in METRIC_NAMES}
