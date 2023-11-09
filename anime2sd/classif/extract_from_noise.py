import numpy as np
import logging
from tqdm import tqdm
from typing import Optional

from imgutils.metrics import ccip_difference
from imgutils.metrics import ccip_default_threshold


def classify_characters_imagewise(
    imgs,
    ref_images=None,
    ref_labels=None,
    batch_diff=None,
    batch_same=None,
    cluster_labels=None,
    same_threshold=0.9,
    default_threshold_ratio=0.75,
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
        ref_images (list):
            The list of reference images
        ref_labels (list):
            Labels of the reference images
        batch_diff (array):
            The n times n array for the ccip difference of all the processed images
        batch_same (arrat):
            The n times n array for the ccip same of all the processed images
        cluster_labels (list):
            Labels of the cluster samples
        same_threshold (float):
            The threshold on same character image proportion to determine whether
            a image belongs to a cluster or not
        default_thresohold_ratio (float):
            The ratio of the default threshold to determine whether two images
            represent the same character or not
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
                ccip_default_threshold() * default_threshold_ratio
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
    images: np.ndarray,
    labels: np.ndarray,
    batch_diff: np.ndarray,
    batch_same: np.ndarray,
    characters_per_image: Optional[np.ndarray] = None,
    ref_images: Optional[np.ndarray] = None,
    ref_labels: Optional[np.ndarray] = None,
) -> None:
    """
    Classify characters for images that do not belong to any cluster (labeled as noise).
    Reference images and characters_per_image may be used as well for classification.
    It updates the labels array in place for noise images with their classified labels.

    Args:
        images (np.ndarray):
            ccip embeddinggs of all processed images.
        labels (np.ndarray):
            Labels of the processed images.
        batch_diff (np.ndarray):
            The n x n array for the ccip difference of all the processed images.
        batch_same (np.ndarray):
            The n x n boolean array indicating similarity below a threshold.
        characters_per_image (Optional[np.ndarray]):
            An optional boolean array (num_images x num_characters)
            indicating the presence of characters in each image. Defaults to None.
        ref_images (Optional[np.ndarray]):
            Optional array of ccip embeddings of the reference images. Defaults to None.
        ref_labels (Optional[np.ndarray]):
            Optional array of labels for the reference images. Defaults to None.

    Returns:
        None: The function performs in-place operations on the
              labels array and does not return a value.
    """
    noise_indices = np.where(labels == -1)[0]
    images_noise = images[noise_indices]
    batch_diff_noise = batch_diff[noise_indices]
    batch_same_noise = batch_same[noise_indices]

    noise_new_labels = classify_characters_imagewise(
        images_noise,
        ref_images=ref_images,
        ref_labels=ref_labels,
        batch_diff=batch_diff_noise,
        batch_same=batch_same_noise,
        cluster_labels=labels,
        same_threshold=0.9,
        default_threshold_ratio=0.5,
        desc="Matching for noises",
    )

    for idx, new_label in zip(noise_indices, noise_new_labels):
        if new_label != -1:
            labels[idx] = new_label
        elif characters_per_image is not None:
            # Update label based on characters_per_image
            # if there's only one corresponding character
            character_counts = characters_per_image[idx]
            if np.sum(character_counts) == 1:
                labels[idx] = np.argmax(character_counts)

    logging.info("Noise extracting complete.")
    label_cnt = {
        i: (labels == i).sum() for i in np.unique(labels) if (labels == i).sum() > 0
    }
    logging.info(f"Current label count: {label_cnt}")
