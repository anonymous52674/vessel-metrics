"""Pixel-based metrics calculated from the confusion matrix or from probability maps."""

import numpy as np
from sklearn import metrics


def confusion_matrix_metrics(pred, target):
    """Calculate accuracy, precision, recall, Dice and IoU between two binary masks.

    Returns
    -------
        A dictionary with keys ACC, PRE, REC, DIC and IOU.
    """

    eps = np.finfo(np.float32).eps

    pred = pred.ravel()
    target = target.ravel()
    tp = np.sum(target & pred)
    tn = np.sum(~target & ~pred)
    fp = np.sum(~target & pred)
    fn = np.sum(target & ~pred)

    return {
        "ACC": (tp + tn) / (tp + tn + fp + fn + eps),
        "PRE": tp / (tp + fp + eps),
        "REC": tp / (tp + fn + eps),
        "DIC": 2 * tp / (2 * tp + fp + fn + eps),
        "IOU": tp / (tp + fp + fn + eps),
    }


def matthews_corrcoef(pred, target):
    """Calculate the Matthews correlation coefficient between two binary masks.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    return metrics.matthews_corrcoef(target.ravel(), pred.ravel())


def roc_auc(prob, target):
    """Calculate the area under the ROC curve from a probability map. Returns nan if the target
    contains a single class.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """

    if target.all() or not target.any():
        return np.nan
    return metrics.roc_auc_score(target.ravel(), prob.ravel())


def average_precision(prob, target):
    """Calculate the average precision from a probability map. Returns nan if the target does
    not contain foreground pixels.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    """

    if not target.any():
        return np.nan
    return metrics.average_precision_score(target.ravel(), prob.ravel())
