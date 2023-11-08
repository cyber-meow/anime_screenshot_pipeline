import numpy as np
import logging
from tqdm import tqdm

from imgutils.metrics import ccip_difference
from imgutils.metrics import ccip_default_threshold


def classify_characters_imagewise(
    imgs,
    same_threshold=0.9,
    default_thresohold_ratio=0.75,
    ref_labels=None,
    cluster_labels=None,
    ref_images=None,
    batch_diff=None,
    batch_same=None,
    desc=None,
):
    """Classify the images one by one with the help of either the reference images
    or the cluster samples
    Their labels should be different to avoid collision
    For reference images, only one image is similar is sufficient
    For cluster samples, we require a certain proportion of images to be similar

    Args:
        imgs (list):
            A list of images to be classified
        same_threshold (float):
            The threshold on same character image proportion to determine whether
            a image belongs to a cluster or not
        default_thresohold_ratio (float):
            The ratio of the default threshold to determine whether two images
            represent the same character or not
        ref_labels (list):
            Labels of the reference images
        cluster_labels (list):
            Labels of the cluster samples
        ref_images (list):
            The list of reference images
        batch_diff (array):
            The n times n array for the ccip difference of all the processed images
        batch_same (arrat):
            The n times n array for the ccip same of all the processed images
        desc (text):
            Description text for tqdm

    Returns:
        The class labels of the images to be classified
    """
    assert ref_images is not None or batch_diff is not None

    cls_labels = -1 * np.ones(len(imgs))
    cls_labels = -1 * np.ones(len(imgs))
    max_ref_label = np.max(ref_labels) if ref_labels is not None else -1
    max_cluster_label = np.max(cluster_labels) if cluster_labels is not None else -1

    for i, img in tqdm(enumerate(imgs), desc=desc):
        if ref_images is not None:
            ref_diff = np.array(
                [ccip_difference(img, ref_img) for ref_img in ref_images]
            )
            ref_avg_dists = np.array(
                [
                    ref_diff[ref_labels == label].min()
                    if len(ref_diff[ref_labels == label]) > 0
                    else np.inf
                    for label in range(0, max_ref_label + 1)
                ]
            )
            batch_same_ref = ref_diff <= (
                ccip_default_threshold() * default_thresohold_ratio
            )
            ref_r_sames = np.array(
                [
                    batch_same_ref[ref_labels == label].mean()
                    for label in range(0, max_ref_label + 1)
                ]
            )
        if batch_diff is not None:
            batch_diff_i = batch_diff[i]
            cluster_avg_dists = np.array(
                [
                    batch_diff_i[cluster_labels == label].mean()
                    if len(batch_diff_i[cluster_labels == label]) > 0
                    else np.inf
                    for label in range(0, max_cluster_label + 1)
                ]
            )
            batch_same_i = batch_same[i]
            # May get nan here for some empty slice
            r_sames = np.array(
                [
                    batch_same_i[cluster_labels == label].mean()
                    for label in range(0, max_cluster_label + 1)
                ]
            )
            if ref_images is not None and ref_avg_dists.min() < cluster_avg_dists.min():
                best_id = np.argmin(ref_avg_dists)
                # For reference images only one is similar is enough
                if ref_r_sames[best_id] > 0:
                    cls_labels[i] = best_id
                else:
                    best_id = np.argmin(cluster_avg_dists)
                    if r_sames[best_id] > same_threshold:
                        cls_labels[i] = best_id
            else:
                best_id = np.argmin(cluster_avg_dists)
                if r_sames[best_id] > same_threshold:
                    cls_labels[i] = best_id
        else:
            best_id = np.argmin(ref_avg_dists)
            if ref_r_sames[best_id] > 0:
                cls_labels[i] = best_id

    return cls_labels


def extract_from_noise(
    images, labels, batch_diff, batch_same, ref_images=None, ref_labels=None
):
    """Classify character for images that do not belong to any cluster
    Rerefence images may be used as well
    Their labels should be different to avoid collision

    Args:
        images (list): List of all processed images
        labels (list): Labels of the processed images
        batch_diff (array):
            The n times n array for the ccip difference of all the processed images
        batch_same (array):
            The n times n array for the ccip same of all the processed images
        ref_images (list): List of reference images
        ref_labels (list): Labels of the reference images

    Returns:
        None: The function performs in-place operations and does not return a value.
    """
    images_noise = images[labels == -1]
    noise_new_labels = classify_characters_imagewise(
        images_noise,
        ref_images=ref_images,
        ref_labels=ref_labels,
        batch_diff=batch_diff[labels == -1],
        batch_same=batch_same[labels == -1],
        cluster_labels=labels,
        default_thresohold_ratio=0.5,
        same_threshold=0.9,
        desc="Matching for noises",
    )
    labels[labels == -1] = noise_new_labels

    logging.info("Noise extracting complete.")
    label_cnt = {
        i: (labels == i).sum()
        for i in range(-1, max(labels) + 1)
        if (labels == i).sum() > 0
    }
    logging.info(f"Current label count: {label_cnt}")
