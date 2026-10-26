"""Metrics based on the centerlines and on the topology of the segmentations."""

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

from .skeleton_graph import create_graph


def cl_dice(pred, target):
    """Calculate the centerline Dice between two binary masks.
    https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html
    """

    pred_skel = skeletonize(pred)
    target_skel = skeletonize(target)

    tprec = (pred_skel & target).sum() / pred_skel.sum() if pred_skel.any() else 0.
    tsens = (target_skel & pred).sum() / target_skel.sum() if target_skel.any() else 0.

    if tprec + tsens == 0:
        return 0.
    return 2 * tprec * tsens / (tprec + tsens)


def _skeleton_length_map(skel):

    skel = skel.astype(np.uint8)
    axial = ndi.correlate(skel, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8), mode="constant")
    diagonal = ndi.correlate(skel, np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.uint8), mode="constant")

    return 0.5 * (axial + np.sqrt(2) * diagonal) * skel


def _buffer(skel, width):

    if not skel.any():
        return np.zeros_like(skel, dtype=bool)
    return ndi.distance_transform_edt(~skel) <= width


def wiedemann_metrics(pred, target, buffer_width=3.):
    """Completeness, correctness and quality as defined in Wiedemann et al., "Empirical
    evaluation of automatically extracted road axes" (1998). The centerlines of the masks are
    obtained by skeletonization and matched using the buffer method.

    Returns
    -------
        A dictionary with keys COM (completeness), COR (correctness) and QUA (quality).
    """

    pred_skel = skeletonize(pred)
    target_skel = skeletonize(target)
    pred_length = _skeleton_length_map(pred_skel)
    target_length = _skeleton_length_map(target_skel)

    total_pred = pred_length.sum()
    total_target = target_length.sum()
    matched_pred = pred_length[_buffer(target_skel, buffer_width)].sum()
    matched_target = target_length[_buffer(pred_skel, buffer_width)].sum()
    unmatched_target = total_target - matched_target

    completeness = matched_target / total_target if total_target > 0 else 0.
    correctness = matched_pred / total_pred if total_pred > 0 else 0.
    den = total_pred + unmatched_target
    quality = matched_pred / den if den > 0 else 0.

    return {"COM": completeness, "COR": correctness, "QUA": quality}


def _add_weights(graph):
    """Calculates the path length of each edge in `graph` and adds it as an edge attribute."""

    weights = {}
    for n1, n2, key in graph.edges:
        distance = 0.
        path = graph.get_edge_data(n1, n2, key)['path']
        for i in range(1, len(path)):
            pix1 = path[i-1]
            pix2 = path[i]
            distance += np.sqrt((pix2[0]-pix1[0])**2 + (pix2[1]-pix1[1])**2)
        weights[(n1, n2, key)] = distance

    nx.set_edge_attributes(graph, weights, 'weight')


def _get_all_pairs_of_nodes(nodes):

    pairs = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            pairs.append((nodes[i], nodes[j]))

    return pairs


def _mask_to_graph(mask):

    skel = skeletonize(mask)
    if not skel.any():
        return nx.MultiGraph()
    graph = create_graph(skel.astype(np.uint8))
    _add_weights(graph)

    return graph


def apls(pred, target, node_threshold=12):
    """APLS: Average Path-Length Similarity. Calculates the differences between the shortest
    distances between pairs of nodes of the ground truth and predicted graphs.

    https://arxiv.org/abs/1807.01232
    """

    gt_graph = _mask_to_graph(target)
    predicted_graph = _mask_to_graph(pred)

    all_nodes = list(gt_graph.nodes)
    if len(all_nodes) == 0:
        return 0.0
    if len(all_nodes) <= node_threshold:
        nodes_to_calculate = all_nodes
    else:
        idxs = np.linspace(0, len(all_nodes) - 1, node_threshold, dtype=int)
        nodes_to_calculate = [all_nodes[i] for i in idxs]

    pred_nodes = list(predicted_graph.nodes)
    pred_positions = np.array([predicted_graph.nodes[n]['center'] for n in pred_nodes])

    def nearest_pred_node(gt_pos):
        dists = np.linalg.norm(pred_positions - np.array(gt_pos), axis=1)
        return pred_nodes[int(np.argmin(dists))]

    C_values = []
    for start, end in _get_all_pairs_of_nodes(nodes_to_calculate):
        if not nx.has_path(gt_graph, start, end):
            continue

        gt_min_distance = nx.shortest_path_length(gt_graph, start, end, weight='weight')
        if gt_min_distance == 0:
            continue

        if len(pred_nodes) == 0:
            C_values.append(1.0)
            continue

        matched_start = nearest_pred_node(gt_graph.nodes[start]['center'])
        matched_end = nearest_pred_node(gt_graph.nodes[end]['center'])

        if not nx.has_path(predicted_graph, matched_start, matched_end):
            C_values.append(1.0)
            continue

        pred_min_distance = nx.shortest_path_length(
            predicted_graph, matched_start, matched_end, weight='weight')

        C = min(1.0, abs(gt_min_distance - pred_min_distance) / gt_min_distance)
        C_values.append(C)

    if not C_values:
        return 0.0
    return 1 - float(np.mean(C_values))
