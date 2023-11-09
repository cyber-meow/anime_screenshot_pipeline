import numpy as np
import logging
from tqdm import tqdm
from typing import Optional, List

from imgutils.metrics import ccip_difference
from imgutils.metrics import ccip_default_threshold


def classify_characters_imagewise(
    img_files: List[str],
    imgs: np.ndarray,
    ref_images: Optional[np.ndarray] = None,
    ref_labels: Optional[np.ndarray] = None,
    batch_diff: Optional[np.ndarray] = None,
    batch_same: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    characters_per_image: Optional[np.ndarray] = None,
    same_threshold_rel: float = 0.6,
    same_threshold_abs: int = 10,
    default_threshold_ratio: float = 0.75,
    to_filter: bool = False,
):
    """
    Classify the images one by one with the help of either the reference images
    or the cluster samples.
    Their labels should be different to avoid collision.
    For reference images, only one image is similar is sufficient.
    For cluster samples, we require a certain proportion and number of images
    to be similar; satisfying one of them is sufficient.

    Args:
        img_files (list):
            Paths to the images to be classified.
        imgs (np.ndarray):
            CCIP embeddings of images to be classified.
        ref_images (np.ndarray, optional):
            The list of reference images.
        ref_labels (np.ndarray, optional):
            Labels of the reference images.
        batch_diff (np.ndarray, optional):
            The n times m array for the ccip difference of the images.
        batch_same (np.ndarray, optional):
            The n times m array for the ccip same of the images.
        cluster_labels (np.ndarray, optional):
            Labels of the cluster samples.
        characters_per_image (np.ndarray, optional):
            An optional boolean array (num_images x num_characters)
            indicating the presence of characters in each image. Defaults to None.
        same_threshold_rel (float, optional):
            The relative threshold for determining whether images belong
            to the same cluster. Defaults to 0.6.
        same_threshold_abs (int, optional):
            The absolute threshold for determining whether images belong
            to the same cluster. Defaults to 10.
        to_filter (bool, optional):
            Only perform filtering for classified images instead of classification
            Assume that cluster_labels are exactly the labels of imgs.

    Returns:
        The class labels of the images to be classified
    """
    assert ref_images is not None or batch_diff is not None

    cls_labels = -1 * np.ones(len(imgs))
    max_ref_label = np.max(ref_labels) if ref_labels is not None else -1
    max_cluster_label = np.max(cluster_labels) if cluster_labels is not None else -1

    for i, img in tqdm(enumerate(imgs)):
        if ref_images is not None and not to_filter:
            # Find the best label from ref_images
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
        if batch_diff is not None:
            batch_diff_i = batch_diff[i]
            ref_label_selected = False
            # Use the original label if we only want to perform filtering
            if to_filter:
                best_id = cluster_labels[i]
                if best_id == -1:
                    continue
            # Find the best label from batch_diff
            else:
                cluster_avg_dists = np.array(
                    [
                        batch_diff_i[cluster_labels == label].mean()
                        if len(batch_diff_i[cluster_labels == label]) > 0
                        else np.inf
                        for label in range(0, max_cluster_label + 1)
                    ]
                )
                if (
                    ref_images is not None
                    and ref_avg_dists.min() < cluster_avg_dists.min()
                ):
                    best_id = np.argmin(ref_avg_dists)
                    ref_r_same = batch_same_ref[ref_labels == best_id].sum()
                    # For reference images only one is similar is enough
                    # Ensure consistent with existing metadata if any
                    if ref_r_same > 0 and (
                        characters_per_image is None or characters_per_image[i, best_id]
                    ):
                        cls_labels[i] = best_id
                        ref_label_selected = True
                if not ref_label_selected:
                    best_id = np.argmin(cluster_avg_dists)
            if not ref_label_selected:
                # Make sure it is consistent with existing metadata
                if (
                    characters_per_image is not None
                    and best_id < characters_per_image.shape[1]
                    and not characters_per_image[i, best_id]
                ):
                    continue
                r_same_mean = batch_same[i][cluster_labels == best_id].mean()
                r_same_sum = batch_same[i][cluster_labels == best_id].sum()
                # if best_id == cluster_labels[i] == 3:
                #     logging.info(str(img_files[i]))
                #     logging.info(str(best_id))
                #     logging.info(str(r_same_mean))
                #     logging.info(str(r_same_sum))
                if (
                    r_same_mean >= same_threshold_rel
                    or r_same_sum >= same_threshold_abs
                ):
                    cls_labels[i] = best_id
        else:
            best_id = np.argmin(ref_avg_dists)
            ref_r_same = batch_same_ref[ref_labels == best_id].sum()
            if ref_r_same > 0 and (
                characters_per_image is None or characters_per_image[i, best_id]
            ):
                cls_labels[i] = best_id

    return cls_labels


def extract_from_noise(
    img_files: List[str],
    images: np.ndarray,
    labels: np.ndarray,
    batch_diff: np.ndarray,
    batch_same: np.ndarray,
    characters_per_image: Optional[np.ndarray] = None,
    ref_images: Optional[np.ndarray] = None,
    ref_labels: Optional[np.ndarray] = None,
    same_threshold_rel: float = 0.6,
    same_threshold_abs: int = 10,
) -> None:
    """
    Classify characters for images that do not belong to any cluster (labeled as noise).
    Reference images and characters_per_image may be used as well for classification.
    It updates the labels array in place for noise images with their classified labels.

    Args:
        img_files (list):
            Paths to the images to be classified.
        imgs (np.ndarray):
            CCIP embeddings of images to be classified.
        labels (np.ndarray):
            Labels of the processed images.
        batch_diff (np.ndarray):
            The n x n array for the ccip difference of all the processed images.
        batch_same (np.ndarray):
            The n x n boolean array indicating similarity below a threshold.
        characters_per_image (np.ndarray, optional):
            An optional boolean array (num_images x num_characters)
            indicating the presence of characters in each image. Defaults to None.
        ref_images (np.ndarray, optional):
            Optional array of ccip embeddings of the reference images. Defaults to None.
        ref_labels (np.ndarray, optional):
            Optional array of labels for the reference images. Defaults to None.
        same_threshold_rel (float, optional):
            The relative threshold for determining whether images belong
            to the same cluster. Defaults to 0.6.
        same_threshold_abs (int, optional):
            The absolute threshold for determining whether images belong
            to the same cluster. Defaults to 10.

    Returns:
        None: The function performs in-place operations on the
              labels array and does not return a value.
    """
    noise_indices = np.where(labels == -1)[0]
    images_noise = images[noise_indices]
    batch_diff_noise = batch_diff[noise_indices]
    batch_same_noise = batch_same[noise_indices]

    logging.info("Matching for noises ...")
    noise_new_labels = classify_characters_imagewise(
        img_files,
        images_noise,
        ref_images=ref_images,
        ref_labels=ref_labels,
        batch_diff=batch_diff_noise,
        batch_same=batch_same_noise,
        cluster_labels=labels,
        characters_per_image=characters_per_image,
        same_threshold_rel=same_threshold_rel,
        same_threshold_abs=same_threshold_abs,
        default_threshold_ratio=0.5,
        to_filter=False,
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


def filter_characters_from_images(
    img_files: List[str],
    images: np.ndarray,
    labels: np.ndarray,
    batch_diff: np.ndarray,
    batch_same: np.ndarray,
    same_threshold_rel: float = 0.6,
    same_threshold_abs: int = 10,
) -> np.ndarray:
    """
    Filters out images with low character similarity based on CCIP embeddings.

    Args:
        img_files (list):
            Paths to the images to be classified.
        imgs (np.ndarray):
            CCIP embeddings of images to be classified.
        labels (np.ndarray):
            The current labels assigned to the processed images.
        batch_diff (np.ndarray):
            The pairwise CCIP difference matrix for the processed images.
        batch_same (np.ndarray):
            The pairwise boolean ccip same matrix for the processed images.
        same_threshold_rel (float, optional):
            The relative threshold for determining whether images belong
            to the same cluster. Defaults to 0.6.
        same_threshold_abs (int, optional):
            The absolute threshold for determining whether images belong
            to the same cluster. Defaults to 10.

    Returns:
        np.ndarray: The filtered labels array.
    """

    logging.info("Filtering characters from images ...")
    filtered_labels = classify_characters_imagewise(
        img_files,
        images,
        batch_diff=batch_diff,
        batch_same=batch_same,
        cluster_labels=labels,
        same_threshold_rel=same_threshold_rel,
        same_threshold_abs=same_threshold_abs,
        default_threshold_ratio=1,
        to_filter=True,
    )

    logging.info("Filtering complete.")
    label_cnt = {
        i: (filtered_labels == i).sum()
        for i in np.unique(filtered_labels)
        if (filtered_labels == i).sum() > 0
    }
    logging.info(f"Current label count: {label_cnt}")
    return filtered_labels
