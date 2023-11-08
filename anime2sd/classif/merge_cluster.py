import logging
import numpy as np
from typing import Set, List, Literal

from imgutils.metrics import ccip_difference, ccip_default_threshold


def merge_clusters(
    exist_ids: Set[int],
    max_clu_id: int,
    batch_same: np.ndarray,
    labels: np.ndarray,
    min_merge_id: int = 0,
    merge_threshold: float = 0.85,
) -> None:
    """
    Merges clusters based on a similarity score threshold. Clusters with a similarity
    score above the threshold will be merged. The merging process respects a minimum
    merge ID, below which cluster IDs will not be merged with each other, but can be
    merged into higher IDs.

    Args:
        exist_ids (Set[int]):
            A set of existing cluster IDs.
        max_clu_id (int):
            The maximum cluster ID.
        min_merge_id (int):
            The minimum cluster ID that can be merged with another cluster.
            Defaults to 0.
        batch_same (np.ndarray):
            A square array where element [i, j] represents the whether images i and j
            are thought to contain the same character or not.
        labels (np.ndarray):
            An array where each element is the cluster ID assigned to the
            corresponding image.
        merge_threshold (float):
            The threshold above which two clusters are considered similar enough to be
            merged. Defaults to 0.85.

    Returns:
        None: The function performs in-place operations and does not return a value.
    """
    # Perform in-place operations to merge clusters
    logging.info("Merging clusters ...")
    while True:
        _round_merged = False
        for xi in range(1, max_clu_id + 1):
            if xi not in exist_ids:
                continue
            for yi in range(max(xi + 1, min_merge_id), max_clu_id + 1):
                if yi not in exist_ids:
                    continue

                score = (batch_same[labels == xi][:, labels == yi]).mean()
                logging.info(f"Label {xi} and {yi}'s similarity score: {score}")
                if score >= merge_threshold:
                    labels[labels == yi] = xi
                    logging.info(f"Merging label {yi} into {xi} ...")
                    exist_ids.remove(yi)
                    _round_merged = True

        if not _round_merged:
            break

    logging.info("Merge complete, remained cluster ids: " + f"{sorted(exist_ids)}.")
    label_cnt = {
        i: (labels == i).sum()
        for i in range(-1, max_clu_id + 1)
        if (labels == i).sum() > 0
    }
    logging.info(f"Current label count: {label_cnt}")


def map_clusters_to_reference(
    imgs: np.ndarray,
    ref_images: np.ndarray,
    ref_labels,
    cluster_ids,
    mode: Literal["min", "avg"] = "min",
    same_threshold: float = 0.5,
) -> np.ndarray:
    """Map cluster IDs of images to the labels of reference images based on similarity.

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

    Returns:
        np.ndarray: An array of labels where each label corresponds to the reference
        images that are the most similar to the image cluster.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    logging.info("Classifying ...")

    cls_labels = cluster_ids.copy()
    max_label = np.max(ref_labels)

    unique_clusters = set(cluster_ids)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    for cluster_id in unique_clusters:
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

        r_sames = np.array(
            [
                (avg_cluster_diff <= ccip_default_threshold())[
                    ref_labels == label
                ].mean()
                for label in range(0, max_label + 1)
            ]
        )

        best_id = np.argmin(avg_dists)
        if r_sames[best_id] >= same_threshold:
            cls_labels[cluster_ids == cluster_id] = best_id

    return cls_labels


def map_clusters_to_existing(labels, characters_per_image, min_proportion=0.5):
    pass
