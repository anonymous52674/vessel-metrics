"""Synthetic perturbations applied to binary vessel masks."""

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize


def remove_comps(img_bool, img_spike):
    """Remove connected components from `img_spike` that are not connected to vessels in `img_bool`."""

    img_lab, num_comp = ndi.label(img_spike)
    comps_to_keep = np.unique(img_bool*img_lab)[1:]

    mask = np.zeros(num_comp+1, dtype=bool)
    mask[comps_to_keep] = True

    return mask[img_lab]


def create_spikes(img_bool, spike_p, rng):
    """Create spikes in the image."""

    img_spike = img_bool.copy()

    spike_mask = rng.random(img_bool.shape) < spike_p
    img_spike[~img_bool & spike_mask] = True

    return remove_comps(img_bool, img_spike)


def create_fps(img_bool, fp_p, rng):
    """Create false positives (islands) in the image."""

    img_fp = img_bool.copy()

    s = np.ones((3, 3))
    img_bool_dil = ndi.binary_dilation(img_bool, structure=s)

    fp_mask = rng.random(img_bool.shape) < fp_p
    fp_mask = ~img_bool_dil & fp_mask

    img_fp[~img_bool & fp_mask] = True

    return img_fp


def create_fns(img_bool, fn_p, rng):
    """Create false negatives (holes) in the image."""

    img_fn = img_bool.copy()

    fn_mask = rng.random(img_bool.shape) < fn_p

    img_fn[fn_mask] = False

    return img_fn


def create_noisy_img(img, spike_p, fp_p, fn_p, rng):
    """Create a noisy image with spikes, false positives (islands), and false negatives (holes).

    Parameters
    ----------
    img
        Binary image.
    spike_p
        Probability of adding a spike pixel. Must be between 0 and 1.
    fp_p
        Probability of false positives (islands).
    fn_p
        Probability of false negatives (holes).
    rng
        numpy random generator.

    Returns
    -------
        Noisy image.
    """

    img_bool = img > 0

    img_spike = create_spikes(img_bool, spike_p, rng)
    img_spike_fp = create_fps(img_spike, fp_p, rng)

    return create_fns(img_spike_fp, fn_p, rng)


def make_skeleton_4connected(bg_skel, foreground):
    """
    Converts an 8-connected skeleton into a 4-connected boundary by patching
    only the diagonal gaps, adding exactly one pixel per gap. The added pixel is
    never a foreground pixel.
    """
    tl_br = (
        bg_skel[:-1, :-1] &
        bg_skel[1:, 1:] &
        ~bg_skel[:-1, 1:] &
        ~bg_skel[1:, :-1]
    )

    tr_bl = (
        bg_skel[:-1, 1:] &
        bg_skel[1:, :-1] &
        ~bg_skel[:-1, :-1] &
        ~bg_skel[1:, 1:]
    )

    top_right_free = ~foreground[:-1, 1:]
    bottom_left_free = ~foreground[1:, :-1]
    top_left_free = ~foreground[:-1, :-1]
    bottom_right_free = ~foreground[1:, 1:]

    patch = np.zeros_like(bg_skel)
    patch[:-1, 1:] |= tl_br & top_right_free
    patch[1:, :-1] |= tl_br & ~top_right_free & bottom_left_free
    patch[:-1, :-1] |= tr_bl & top_left_free
    patch[1:, 1:] |= tr_bl & ~top_left_free & bottom_right_free

    return bg_skel | patch


def get_exact_distances(max_dist):
    """
    Generates all unique Euclidean distances between pixels up to a maximum distance.
    """
    max_offset = int(np.ceil(max_dist))
    coords = np.arange(max_offset + 1)
    squared_distances = coords[:, None]**2 + coords[None, :]**2
    distances = np.sqrt(np.unique(squared_distances))

    return distances[(distances > 0) & (distances <= max_dist)]


def border_perturbation(img, max_dist, mode):
    """
    Erodes or dilates a binary blood vessel image while preserving topology.

    Parameters:
        img: 2D binary numpy array (True for vessels, False for background).
        max_dist: The maximum distance by which to erode or dilate.
        mode: "erode" or "dilate".
    Returns:
        3D numpy array containing the sequence of processed binary images.
        1D numpy array containing the distances used.
    """
    if mode not in ["erode", "dilate"]:
        raise ValueError(f"Unknown mode {mode}.")

    img = img.astype(bool)

    pad_size = int(np.ceil(max_dist)) + 1
    padded_img = np.pad(img, pad_width=pad_size, mode='symmetric')
    distances = get_exact_distances(max_dist)

    if mode == "dilate":
        foreground = padded_img
        padded_img = ~padded_img

    dt = ndi.distance_transform_edt(padded_img)

    if mode == "erode":
        img_skel = skeletonize(img)
        skel = np.pad(img_skel, pad_width=pad_size)
    else:
        skel = make_skeleton_4connected(skeletonize(padded_img), foreground)

    imgs = []
    for dist in distances:
        if mode == "erode":
            result = (dt > dist) | skel
            seeds = img_skel
        else:
            result = (dt <= dist) & ~skel
            seeds = img

        result = result[pad_size:-pad_size, pad_size:-pad_size]

        img_lab, _ = ndi.label(result, structure=np.ones((3, 3)))
        result = np.isin(img_lab, np.unique(img_lab[seeds]))

        if len(imgs) > 0 and np.array_equal(result, imgs[-1]):
            break

        imgs.append(result)

    return np.array(imgs), distances[:len(imgs)]


def random_crop(img, rng, size=256, min_vessel_fraction=0.1, max_attempts=1000):
    """Randomly crop a `size` x `size` region containing at least `min_vessel_fraction` of vessel
    pixels. If no such region is found after `max_attempts`, the region with the largest vessel
    fraction is returned. Images smaller than `size` are returned unchanged."""

    img_h, img_w = img.shape
    if img_h <= size or img_w <= size:
        return img

    best_crop, best_fraction = None, -1
    for _ in range(max_attempts):
        h1 = rng.integers(0, img_h - size + 1)
        w1 = rng.integers(0, img_w - size + 1)
        crop = img[h1:h1+size, w1:w1+size]
        fraction = crop.mean()
        if fraction >= min_vessel_fraction:
            return crop
        if fraction > best_fraction:
            best_crop, best_fraction = crop, fraction

    return best_crop
