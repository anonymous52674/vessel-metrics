"""Human Correction Efforts (HCE) metric. Adapted from https://github.com/xuebinqin/DIS"""

import cv2 as cv
import numpy as np
from skimage.measure import label
from skimage.morphology import disk, skeletonize


def filter_bdy_cond(bdy_, mask, cond):

    cond = cv.dilate(cond.astype(np.uint8), disk(1))
    labels = label(mask)
    indep = np.ones(np.unique(labels).shape[0])
    indep[0] = 0

    boundaries = []
    ind_map = np.zeros(cond.shape[0:2])

    for i in range(0, len(bdy_)):
        tmp_bdies = []
        tmp_bdy = []
        for j in range(0, bdy_[i].shape[0]):
            r, c = bdy_[i][j, 0, 1], bdy_[i][j, 0, 0]

            if np.sum(cond[r, c]) == 0 or ind_map[r, c] != 0:
                if len(tmp_bdy) > 0:
                    tmp_bdies.append(tmp_bdy)
                    tmp_bdy = []
                continue
            tmp_bdy.append([c, r])
            ind_map[r, c] = ind_map[r, c] + 1
            indep[labels[r, c]] = 0
        if len(tmp_bdy) > 0:
            tmp_bdies.append(tmp_bdy)

        if len(tmp_bdies) > 1:
            first_x, first_y = tmp_bdies[0][0]
            last_x, last_y = tmp_bdies[-1][-1]
            if ((abs(first_x - last_x) == 1 and first_y == last_y) or
                    (first_x == last_x and abs(first_y - last_y) == 1) or
                    (abs(first_x - last_x) == 1 and abs(first_y - last_y) == 1)):
                tmp_bdies[-1].extend(tmp_bdies[0][::-1])
                del tmp_bdies[0]

        for k in range(0, len(tmp_bdies)):
            tmp_bdies[k] = np.array(tmp_bdies[k])[:, np.newaxis, :]
        if len(tmp_bdies) > 0:
            boundaries.extend(tmp_bdies)

    return boundaries, np.sum(indep)


def approximate_RDP(boundaries, epsilon=1.0):

    pixel_cnt_ = 0
    for boundary in boundaries:
        pixel_cnt_ = pixel_cnt_ + len(cv.approxPolyDP(boundary, epsilon, False))

    return pixel_cnt_


def relax_HCE(gt, rs, gt_ske, relax=5, epsilon=2.0):

    gt = (gt > 0).astype(np.uint8)
    rs = (rs > 0).astype(np.uint8)

    Union = np.logical_or(gt, rs)
    TP = np.logical_and(gt, rs)
    FP = rs - TP
    FN = gt - TP

    Union_erode = cv.erode(Union.astype(np.uint8), disk(1), iterations=relax)

    FP_ = np.logical_and(FP, Union_erode)
    for i in range(0, relax):
        FP_ = cv.dilate(FP_.astype(np.uint8), disk(1))
        FP_ = np.logical_and(FP_, 1 - np.logical_or(TP, FN))
    FP_ = np.logical_and(FP, FP_)

    FN_ = np.logical_and(FN, Union_erode)
    for i in range(0, relax):
        FN_ = cv.dilate(FN_.astype(np.uint8), disk(1))
        FN_ = np.logical_and(FN_, 1 - np.logical_or(TP, FP))
    FN_ = np.logical_and(FN, FN_)
    FN_ = np.logical_or(FN_, np.logical_xor(gt_ske, np.logical_and(TP, gt_ske)))

    ctrs_FP, _ = cv.findContours(FP_.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    bdies_FP, indep_cnt_FP = filter_bdy_cond(ctrs_FP, FP_, np.logical_or(TP, FN_))
    ctrs_FN, _ = cv.findContours(FN_.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    bdies_FN, indep_cnt_FN = filter_bdy_cond(
        ctrs_FN, FN_, 1 - np.logical_or(np.logical_or(TP, FP_), FN_))

    poly_FP_point_cnt = approximate_RDP(bdies_FP, epsilon=epsilon)
    poly_FN_point_cnt = approximate_RDP(bdies_FN, epsilon=epsilon)

    return poly_FP_point_cnt, indep_cnt_FP, poly_FN_point_cnt, indep_cnt_FN


def hce(pred, target, relax=5, epsilon=2.0):
    """Calculate the HCE between a predicted and a target binary mask."""

    gt_ske = skeletonize(target)
    fp_points, fp_indep, fn_points, fn_indep = relax_HCE(
        target.astype(np.uint8), pred.astype(np.uint8), gt_ske, relax=relax, epsilon=epsilon)

    return float(fp_points + fp_indep + fn_points + fn_indep)
