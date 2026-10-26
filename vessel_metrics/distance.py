# Copyright 2018 Google Inc. All Rights Reserved.
"""Contour distance metrics for 2D binary segmentations. Adapted from DeepMind's repository
https://github.com/google-deepmind/surface-distance
"""
import numpy as np
from scipy import ndimage

from . import lookup_tables


def distance_metrics(pred, target, tolerance=0., percent=100., spacing=(1., 1.)):
    """Computes contour distance metrics between a predicted and a target mask.

    - Average surface distance: the average distance from the contour of the target to the
      contour of the prediction (DGP) and vice versa (DPG).
    - Hausdorff distance (HAU): maximum between the `percent` percentiles of the two directed
      distances.
    - Surface overlap: the fraction of the contour of the target that is within `tolerance` of
      the contour of the prediction (OFG) and vice versa (OFP).
    - Surface Dice (SDC): the Dice coefficient computed on the contours of the target and of the
      prediction.

    Parameters
    ----------
    pred
        Binary predicted mask.
    target
        Binary target mask.
    tolerance
        Maximum distance to consider two contours as overlapping. Only used in the surface
        overlap and surface Dice metrics.
    percent
        Percentile of the distances used in the Hausdorff distance. The value 100 corresponds to
        the standard Hausdorff distance.
    spacing
        Pixel spacing along rows and columns.

    Returns
    -------
        A dictionary with keys HAU, DGP, DPG, OFG, OFP and SDC.
    """

    surface_distances = compute_surface_distances(target, pred, spacing)
    dgp, dpg = compute_average_surface_distance(surface_distances)
    ofg, ofp = compute_surface_overlap_at_tolerance(surface_distances, tolerance)

    return {
        "HAU": compute_robust_hausdorff(surface_distances, percent),
        "DGP": dgp,
        "DPG": dpg,
        "OFG": ofg,
        "OFP": ofp,
        "SDC": compute_surface_dice_at_tolerance(surface_distances, tolerance),
    }


def _compute_bounding_box(mask):
    """Computes the bounding box of the mask.

    Returns
    -------
    A tuple:
        - The coordinates of the first point of the bounding box (smallest on all
          axes), or `None` if the mask contains only zeros.
        - The coordinates of the second point of the bounding box (greatest on all
          axes), or `None` if the mask contains only zeros.
    """
    num_dims = len(mask.shape)
    bbox_min = np.zeros(num_dims, np.int64)
    bbox_max = np.zeros(num_dims, np.int64)

    proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return None, None

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    for axis in range(1, num_dims):
        max_over_axes = list(range(num_dims))
        max_over_axes.pop(axis)
        max_over_axes = tuple(max_over_axes)
        proj = np.amax(mask, axis=max_over_axes)
        idx_nonzero = np.nonzero(proj)[0]
        bbox_min[axis] = np.min(idx_nonzero)
        bbox_max[axis] = np.max(idx_nonzero)

    return bbox_min, bbox_max


def _crop_to_bounding_box(mask, bbox_min, bbox_max):
    """Crops a 2D mask to the bounding box specified by `bbox_{min,max}`."""
    cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    cropmask[0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1, bbox_min[1]:bbox_max[1] + 1]

    return cropmask


def _sort_distances_surfels(distances, surfel_areas):
    """Sorts the two list with respect to the tuple of (distance, surfel_area)."""
    sorted_surfels = np.array(sorted(zip(distances, surfel_areas)))
    return sorted_surfels[:, 0], sorted_surfels[:, 1]


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    """Computes closest distances from all contour points to the other contour.

    Finds all contour elements in the ground truth mask `mask_gt` and the predicted mask
    `mask_pred`, computes their length and the distance to the closest point on the other
    contour. It returns two sorted lists of distances together with the corresponding contour
    lengths. If one of the masks is empty, the corresponding lists are empty and all distances
    in the other list are `inf`.

    Parameters
    ----------
    mask_gt
        2D bool array. The ground truth mask.
    mask_pred
        2D bool array. The predicted mask.
    spacing_mm
        2-element list-like structure. Pixel spacing in x0 and x1 directions.

    Returns
    -------
    A dict with:
        "distances_gt_to_pred": 1D array with the distances from all ground truth contour
            elements to the predicted contour, sorted from smallest to largest.
        "distances_pred_to_gt": 1D array with the distances from all predicted contour
            elements to the ground truth contour, sorted from smallest to largest.
        "surfel_areas_gt": 1D array with the length of the ground truth contour elements, in
            the same order as distances_gt_to_pred.
        "surfel_areas_pred": 1D array with the length of the predicted contour elements, in
            the same order as distances_pred_to_gt.
    """
    mask_gt = mask_gt > 0
    mask_pred = mask_pred > 0

    if not mask_gt.ndim == mask_pred.ndim == len(spacing_mm) == 2:
        raise ValueError("The masks must be 2D and spacing_mm must have two elements.")

    neighbour_code_to_surface_area = (
        lookup_tables.create_table_neighbour_code_to_contour_length(spacing_mm))
    kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_2D_KERNEL
    full_true_neighbours = 0b1111

    bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    if bbox_min is None:
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([]),
        }

    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    neighbour_code_map_gt = ndimage.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    borders_gt = ((neighbour_code_map_gt != 0) &
                  (neighbour_code_map_gt != full_true_neighbours))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != full_true_neighbours))

    if borders_gt.any():
        distmap_gt = ndimage.distance_transform_edt(~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.distance_transform_edt(~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.inf * np.ones(borders_pred.shape)

    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    if distances_gt_to_pred.shape != (0,):
        distances_gt_to_pred, surfel_areas_gt = _sort_distances_surfels(
            distances_gt_to_pred, surfel_areas_gt)

    if distances_pred_to_gt.shape != (0,):
        distances_pred_to_gt, surfel_areas_pred = _sort_distances_surfels(
            distances_pred_to_gt, surfel_areas_pred)

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
        "surfel_areas_gt": surfel_areas_gt,
        "surfel_areas_pred": surfel_areas_pred,
    }


def compute_average_surface_distance(surface_distances):
    """Returns the average distance from the ground truth contour to the predicted contour and
    the average distance from the predicted contour to the ground truth contour. The length of
    each contour element is taken into account.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = (
        np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt))
    average_distance_pred_to_gt = (
        np.sum(distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred))
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_robust_hausdorff(surface_distances, percent):
    """Computes the robust Hausdorff distance, which uses the `percent` percentile of the
    distances instead of the maximum distance. The percentage is computed by taking the length
    of each contour element into account.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:
        surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[min(idx, len(distances_gt_to_pred) - 1)]
    else:
        perc_distance_gt_to_pred = np.inf

    if len(distances_pred_to_gt) > 0:
        surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
        idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[min(idx, len(distances_pred_to_gt) - 1)]
    else:
        perc_distance_pred_to_gt = np.inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
    """Computes the fraction of the ground truth contour that is within `tolerance_mm` of the
    predicted contour and vice versa. The length of each contour element is taken into account.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    rel_overlap_gt = (
        np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt))
    rel_overlap_pred = (
        np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) /
        np.sum(surfel_areas_pred))
    return (rel_overlap_gt, rel_overlap_pred)


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    """Computes the surface Dice coefficient at a specified tolerance. A contour element is
    counted as overlapping when the closest distance to the other contour is less than or equal
    to `tolerance_mm`.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
        np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice
