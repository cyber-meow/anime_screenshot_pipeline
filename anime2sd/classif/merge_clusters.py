import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Sequence, Literal
from tqdm import tqdm

from imgutils.metrics import ccip_difference, ccip_default_threshold


def assign_if_consistent(
    labels: np.ndarray,
    candidate_labels: Sequence[int],
    characters_per_image: Optional[np.ndarray],
    image_indices: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Assign new labels to images if they are consistent with characters_per_image.
    If characters_per_image is None, it assigns labels without consistency checks.

    Args:
        labels (np.ndarray):
            Current labels for each image.
        candidate_labels (Sequence[int]):
            Candidate labels that could be assigned to the images.
        characters_per_image (Optional[np.ndarray]):
            A boolean array indicating the presence of characters in each image,
            or None if no character information is available.
        image_indices (Optional[np.ndarray]):
            Indices of images to be updated. If None, all images are considered.

    Returns:
        Dict[int, np.ndarray]:
            A dictionary mapping candidate labels to the indices of images that
            are updated to that label.
            Note that the function also updates the labels array in place.
    """

    updated_indices = dict()

    if image_indices is None:
        image_indices = np.arange(labels.size)

    if characters_per_image is None:
        # If no character information is available, assign the candidate label directly
        labels[image_indices] = candidate_labels[0]
        updated_indices[candidate_labels[0]] = image_indices
    else:
        # If characters_per_image is provided, perform consistency checks
        # Initialize an array to keep track of which images have been updated
        updated = np.zeros(labels.shape, dtype=bool)

        for candidate_label in candidate_labels:
            if candidate_label >= characters_per_image.shape[1]:
                indices_to_update = image_indices[~updated[image_indices]]
                labels[indices_to_update] = candidate_label
                updated_indices[candidate_label] = indices_to_update
                break
            # Get the indices of images where the candidate label is consistent and
            # have not been updated yet
            consistent_indices = image_indices[
                (characters_per_image[image_indices, candidate_label])
                & (~updated[image_indices])
            ]
            # Update the labels for these images
            labels[consistent_indices] = candidate_label
            updated_indices[candidate_label] = consistent_indices
            # Mark these images as updated
            updated[consistent_indices] = True
    return updated_indices


def merge_clusters(
    labels: np.ndarray,
    batch_same: np.ndarray,
    min_merge_id: int = 0,
    merge_threshold: float = 0.85,
    characters_per_image: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Merges clusters based on a similarity score threshold. Clusters with a similarity
    score above the threshold will be merged. The merging process respects a minimum
    merge ID, below which cluster IDs will not be merged with each other, but can be
    merged into higher IDs.

    Args:
        labels (np.ndarray):
            An array where each element is the cluster ID assigned to the
            corresponding image.
        batch_same (np.ndarray):
            A square array where element [i, j] represents the whether images i and j
            are thought to contain the same character or not.
        min_merge_id (int):
            The minimum cluster ID that can be merged with another cluster.
            Defaults to 0.
        merge_threshold (float):
            The threshold above which two clusters are considered similar enough to be
            merged. Defaults to 0.85.
        characters_per_image (np.ndarray):
            A boolean array (num_images x num_characters) indicating the presence of
            characters in each image.
        logger (Optional[logging.Logger]):
            A logger to use for logging. Defaults to None, in which case
            the default logger will be used.

    Returns:
        None: The function performs in-place operations and does not return a value.
    """
    if logger is None:
        logger = logging.getLogger()
    exist_ids = np.unique(labels[labels >= 0])
    max_clu_id = np.max(exist_ids)
    exist_ids = set(exist_ids)
    # Perform in-place operations to merge clusters
    logger.info("Merging clusters ...")
    while True:
        round_merged = False
        for xi in range(1, max_clu_id + 1):
            if xi not in exist_ids:
                continue
            for yi in range(max(xi + 1, min_merge_id), max_clu_id + 1):
                if yi not in exist_ids:
                    continue

                score = (batch_same[labels == xi][:, labels == yi]).mean()
                logger.info(f"Label {xi} and {yi}'s similarity score: {score}")
                if score >= merge_threshold:
                    image_indices_to_update = np.where(labels == yi)[0]
                    assign_if_consistent(
                        labels=labels,
                        candidate_labels=[xi],
                        characters_per_image=characters_per_image,
                        image_indices=image_indices_to_update,
                    )
                    logger.info(f"Merging label {yi} into {xi} ...")
                    exist_ids.remove(yi)
                    round_merged = True

        if not round_merged:
            break

    logger.info("Merge complete, remained cluster ids: " + f"{sorted(exist_ids)}.")
    label_cnt = {
        i: (labels == i).sum()
        for i in range(-1, max_clu_id + 1)
        if (labels == i).sum() > 0
    }
    logger.info(f"Current label count: {label_cnt}")


def map_clusters_to_reference(
    imgs: np.ndarray,
    ref_images: np.ndarray,
    ref_labels: np.ndarray,
    cluster_ids: np.ndarray,
    mode: Literal["min", "avg"] = "min",
    same_threshold: float = 0.5,
    characters_per_image: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, Dict[int, List[np.ndarray]]]:
    """
    Map cluster IDs of images to the labels of reference images based on similarity.

    This function classifies each cluster of images by comparing it to a set of
    reference images and assigning the reference label that best matches the cluster
    based on the specified mode.

    Args:
        imgs (List[np.ndarray]):
            ccip embeddings of images to be classify.
        ref_images (List[np.ndarray]):
            ccip embeddings of reference images.
        ref_labels (np.ndarray):
            Labels of the reference images.
        cluster_ids (np.ndarray):
            An array of initial cluster IDs assigned to each image in `imgs`.
        mode (Literal["min", "avg"]):
            The mode to calculate the similarity metric.
            If "min", the minimum distance is used;
            if "avg", the average distance is used. Defaults to "min".
        same_threshold (float):
            The threshold on same character image proportion for determining if a
            cluster is considered the same as a reference.
            Defaults to 0.5.
        characters_per_image (np.ndarray):
            A boolean array (num_images x num_characters) indicating the presence of
            characters in each image.
        logger (Optional[Logger]):
            A logger to use for logging. Defaults to None, in which case
            the default logger will be used.

    Returns:
        np.ndarray:
            An array of labels where each label corresponds to the reference
            images that are the most similar to the image cluster.
        Dict[int, List[np.ndarray]]:
            A dictionary mapping the reference labels to a list of image indices
            that are assigned to the corresponding reference label.
            The lists correspond to indices from different clusters.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info("Classifying using reference images ...")

    updated_labels = cluster_ids.copy()
    updated_indices_mapping = dict()
    max_label = np.max(ref_labels)

    unique_clusters = set(cluster_ids)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    for cluster_id in tqdm(unique_clusters):
        cluster_indices = [
            idx for idx, img in enumerate(imgs) if cluster_ids[idx] == cluster_id
        ]

        cluster_imgs = [imgs[idx] for idx in cluster_indices]
        avg_cluster_diff = np.mean(
            [
                np.array([ccip_difference(img, ref_img) for ref_img in ref_images])
                for img in cluster_imgs
            ],
            axis=0,
        )

        if mode == "avg":
            avg_dists = np.array(
                [
                    avg_cluster_diff[ref_labels == label].mean()
                    if len(avg_cluster_diff[ref_labels == label]) > 0
                    else np.inf
                    for label in range(0, max_label + 1)
                ]
            )
        elif mode == "min":
            avg_dists = np.array(
                [
                    avg_cluster_diff[ref_labels == label].min()
                    if len(avg_cluster_diff[ref_labels == label]) > 0
                    else np.inf
                    for label in range(0, max_label + 1)
                ]
            )
        else:
            raise ValueError("Invalid mode. Choose either 'avg' or 'min'.")

        best_id = np.argmin(avg_dists)
        r_same = (avg_cluster_diff <= ccip_default_threshold())[
            ref_labels == best_id
        ].mean()

        if r_same >= same_threshold:
            image_indices_to_update = np.where(cluster_ids == cluster_id)[0]
            updated_indices = assign_if_consistent(
                labels=updated_labels,
                candidate_labels=[best_id],
                characters_per_image=characters_per_image,
                image_indices=image_indices_to_update,
            )
            # Actually the dictionary contains at most one element and would then
            # be with key best_id
            for label, indices in updated_indices.items():
                if label not in updated_indices_mapping:
                    updated_indices_mapping[label] = []
                updated_indices_mapping[label].append(indices)

    logger.info(
        "Classifying complete, remained cluster ids: "
        + f"{sorted(np.unique(updated_labels))}."
    )

    return updated_labels, updated_indices_mapping


def map_clusters_to_existing(
    labels: np.ndarray,
    characters_per_image: np.ndarray,
    n_pre_labels: int,
    min_proportion: float = 0.6,
    accept_multiple_candidates: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, Dict[int, List[np.ndarray]]]:
    """Maps cluster labels to the most frequent character ID in characters_per_image,
    ensuring that the character ID meets a minimum proportion within the cluster.

    Args:
        labels (np.ndarray):
            An array of integer labels for each image.
        characters_per_image (np.ndarray):
            A boolean array (num_images x num_characters) indicating the presence of
            characters in each image.
        n_pre_labels (int):
            The number of pre-defined labels (from metadata and reference images).
        min_proportion (float):
            The minimum proportion for the most frequent character ID to be considered
            as the representative for the cluster.
        accept_multiple_candidates (bool):
            Whether we try to perform classification when there are multiple
            candidate labels. Defaults to False.
        logger (logging.Logger):
            A logger to use for logging. Defaults to None, in which case
            the default logger will be used.

    Returns:
        np.ndarray:
            An array of updated labels for each image. Labels are only updated
            for clusters where a character ID meets the minimum proportion.
            Otherwise, they remain unchanged.
        Dict[int, List[np.ndarray]]:
            A dictionary mapping the labels from `characters_per_image` to a list of
            image indices that are assigned to the corresponding label.
            The arrays in the lists correspond to indices from different clusters.

    Raises:
        Warning: If there are multiple character IDs that can represent a cluster.
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info("Classifying using existing metadata characters ...")

    updated_labels = labels.copy()  # Create a copy of the labels to update
    updated_indices_mapping = dict()
    unique_labels = np.unique(labels)

    for label in tqdm(unique_labels):
        # Skip if the label is -1 or belongs to predfined class
        if label < n_pre_labels:
            continue

        # Find indices of images in the current cluster
        cluster_indices = np.nonzero(labels == label)[0]
        if cluster_indices.size == 0:
            continue  # No images found for this label

        # Sum presence of each character in the cluster
        character_sums = characters_per_image[cluster_indices].sum(axis=0)
        max_count = np.max(character_sums)
        proportion = max_count / len(cluster_indices)

        if proportion < min_proportion:
            continue  # No character meets the minimum proportion

        # Find character(s) with the maximum count
        max_characters = np.nonzero(character_sums == max_count)[0]
        if len(max_characters) > 1:
            # Warn if there are multiple characters with the same count
            logger.warning(
                f"Label {label} has multiple potential representative characters: "
                f"{max_characters}."
            )
            if not accept_multiple_candidates:
                continue

        # Update the label of the cluster with the character ID that
        # has the maximum count, but only update when agree with orignal labels
        updated_indices = assign_if_consistent(
            labels=updated_labels,
            candidate_labels=max_characters,
            characters_per_image=characters_per_image,
            image_indices=cluster_indices,
        )
        for label, indices in updated_indices.items():
            if label not in updated_indices_mapping:
                updated_indices_mapping[label] = []
            updated_indices_mapping[label].append(indices)

    logger.info(
        "Classifying complete, remained cluster ids: "
        + f"{sorted(np.unique(updated_labels))}."
    )

    return updated_labels, updated_indices_mapping
