import logging
from typing import List, Tuple, Dict, Optional
from hbutils.string import plural_word

import numpy as np
from sklearn.cluster import OPTICS

from imgutils.metrics import ccip_default_threshold
from imgutils.metrics import ccip_batch_differences

from ..basics import remove_empty_folders
from ..character import Character

from .file_utils import load_image_features_and_characters, parse_ref_dir, save_to_dir
from .imagewise import extract_from_noise, filter_characters_from_images
from .merge_clusters import (
    merge_clusters,
    map_clusters_to_existing,
    map_clusters_to_reference,
)


def cluster_characters_basics(
    images: np.ndarray,
    clu_min_samples: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform clustering on a set of images to group similar characters
    using the OPTICS algorithm.

    Args:
        images (np.ndarray):
            ccip embeddings of images on which clustering is to be performed.
        clu_min_samples (int):
            The number of samples in a neighborhood for a point to be considered as
            a core point.
        logger (Optional[Logger]):
            A logger to use for logging. Defaults to None, in which case
            the default logger will be used.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - labels (np.ndarray): An array of cluster labels assigned to each image.
        - batch_diff (np.ndarray):
            A square array of pairwise ccip differences between images.
        - batch_same (np.ndarray):
            A boolean array indicating if pairwise differences are below the threshold.

    """
    if logger is None:
        logger = logging.getLogger()
    batch_diff = ccip_batch_differences(images)
    batch_same = batch_diff <= ccip_default_threshold()

    # clustering
    def _metric(x, y):
        return batch_diff[int(x), int(y)].item()

    logger.info("Clustering ...")
    samples = np.arange(len(images)).reshape(-1, 1)
    # max_eps, _ = ccip_default_clustering_params(method='optics_best')
    clustering = OPTICS(min_samples=clu_min_samples, metric=_metric).fit(samples)
    labels = clustering.labels_.astype(int)

    max_clu_id = labels.max().item()
    all_label_ids = np.array([-1, *range(0, max_clu_id + 1)])
    logger.info(f'Cluster complete, with {plural_word(max_clu_id, "cluster")}.')
    label_cnt = {
        i: (labels == i).sum() for i in all_label_ids if (labels == i).sum() > 0
    }
    logger.info(f"Current label count: {label_cnt}")

    return labels, batch_diff, batch_same


def merge_characters(
    characters: Dict[int, Character],
    ref_characters: Dict[int, Character],
    ref_labels: np.ndarray,
    characters_per_image: Optional[np.ndarray],
) -> Tuple[Dict[int, Character], np.ndarray, Optional[np.ndarray]]:
    """
    Merge reference class names into existing class names, update labels,
    and adjust characters_per_image.

    Args:
        characters (Dict[int, Character]):
            Existing characters with labels as keys.
        ref_characters (Dict[int, Character]):
            Reference characters with labels as keys.
        ref_labels (np.ndarray):
            Array of labels corresponding to reference images.
        characters_per_image (Optional[np.ndarray]):
            Array indicating presence of characters in images.

    Returns:
        Tuple[Dict[int, Character], np.ndarray, Optional[np.ndarray]]:
            - Updated character dictionary.
            - Updated reference labels.
            - Updated characters_per_image if provided.
    """
    # Create a new class_names dictionary to avoid modifying the original one
    updated_characters = characters.copy()
    updated_ref_labels = ref_labels.copy()
    n_existing_labels = max(characters.keys()) + 1 if characters else 0

    # Compute updated ref_labels and characters
    for ref_label, ref_character in ref_characters.items():
        existing_label = next(
            (
                label
                for label, character in characters.items()
                if character == ref_character
            ),
            None,
        )
        if existing_label is not None:
            # Update ref_labels with the existing label from characters
            updated_ref_labels[ref_labels == ref_label] = existing_label
        else:
            # Add new entry to updated_characters if it doesn't exist
            updated_characters[n_existing_labels] = ref_character
            updated_ref_labels[ref_labels == ref_label] = n_existing_labels
            n_existing_labels += 1

    # Update characters_per_image for new labels
    if characters_per_image is not None:
        new_characters_per_image = np.zeros(
            (characters_per_image.shape[0], n_existing_labels), dtype=bool
        )
        new_characters_per_image[
            :, : characters_per_image.shape[1]
        ] = characters_per_image

        # The following assumes that old labels are always correct and new labels
        # are refinement of them
        for new_label in range(characters_per_image.shape[1], n_existing_labels):
            new_character = updated_characters[new_label]

            # Find existing labels whose character name matches the new class
            for label, character in characters.items():
                if character.character_name == new_character.character_name:
                    new_characters_per_image[:, new_label] |= new_characters_per_image[
                        :, label
                    ]

        characters_per_image = new_characters_per_image

    return updated_characters, updated_ref_labels, characters_per_image


def select_indices_recursively(
    list_of_indices: List[np.ndarray], n_to_select: int
) -> List[int]:
    """
    Recursively select indices from a list of np.ndarrays, trying to evenly
    distribute the selection.

    Args:
        list_of_indices (List[np.ndarray]): List of np.ndarrays containing indices.
        n_to_select (int): Total number of indices to select.

    Returns:
        List[int]: List of selected indices.
    """
    if n_to_select <= 0 or not list_of_indices:
        return []

    selected_indices = []
    total_selected = 0

    max_per_array = max(n_to_select // len(list_of_indices), 1)

    # Select indices from each array
    for indices in list_of_indices:
        n_select = min(max_per_array, len(indices), n_to_select - total_selected)
        selected_indices.extend(indices[:n_select])
        total_selected += n_select

    # Recursively select more if needed
    if total_selected < n_to_select:
        remaining_indices = [
            indices[max_per_array:]
            for indices in list_of_indices
            if len(indices) > max_per_array
        ]
        selected_indices.extend(
            select_indices_recursively(remaining_indices, n_to_select - total_selected)
        )

    return selected_indices


def select_to_add_to_ref(
    updated_indices_mapping: Dict[int, List[np.ndarray]],
    n_add_images_to_ref: int,
    shuffle_indices: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select image labels to add to reference images.

    Args:
        updated_indices_mapping (Dict[int, List[np.ndarray]]):
            A dictionary mapping reference or metadata labels to the indices of images
            that are updated to that label.
            The arrays in the lists correspond to indices from different clusters.
        n_add_images_to_ref (int):
            The number of images to add to reference images per character.
        shuffle_indices (bool):
            Whether to shuffle the array indices before selecting them.
            Defaults to True.

    Returns:
        np.array:
            The image indices of images to add to reference directory.
        np.array:
            The image labels of images to add to reference directory.
    """
    selected_indices = []
    labels = []

    for label, list_of_indices in updated_indices_mapping.items():
        if shuffle_indices:
            for indices in list_of_indices:
                np.random.shuffle(indices)
        # Start with an initial maximum per array based on the total number of arrays
        selected_for_label = select_indices_recursively(
            list_of_indices, n_add_images_to_ref
        )
        selected_indices.extend(selected_for_label)
        labels.extend([label] * len(selected_for_label))

    return np.array(selected_indices), np.array(labels)


def classify_from_directory(
    src_dir: str,
    dst_dir: str,
    ref_dir: Optional[str] = None,
    ignore_character_metadata: bool = False,
    to_extract_from_noise: bool = True,
    to_filter: bool = True,
    keep_unnamed: bool = True,
    accept_multiple_candidates: bool = False,
    clu_min_samples: int = 5,
    merge_threshold: float = 0.85,
    same_threshold_rel: float = 0.6,
    same_threshold_abs: int = 10,
    n_add_images_to_ref: int = 0,
    move: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    Classify images from src_dir to dst_dir
    The entire character classification goes through the following process

    1. Extract or load CCIP features from the source directory.
    2. Perform OPTICS clustering to identify clusters of images.
    3. (Optional) Determine labels for images that do not belong to any clusters.
    4. Merge clusters based on similarity. This may involve:
        * Mapping clusters to characters using reference images (if provided).
        * Using image metadata to determine character labels for clusters
          (if available).
        * Merging reamaining clusters based on similarity.
    5. (Optional) Apply a final filtering step to ensure character consistency.

    Args:
        src_dir (str):
            Path to source directory containing images to be classified.
        dst_dir (str):
            Path to destination directory for classified images.
        ref_dir (str):
            Path to reference images. Defaults to None.
        ignore_character_metadata (bool):
            Whether to ignore existing character metadata or not. Defaults to False.
        to_extract_from_noise (bool):
            Whether to perform step 3 (extract from noise) or not. Defaults to True.
        to_filter (bool):
            Whether to perform step 5 (final filtering) or not. Defaults to True.
        keep_unnamed (bool):
            Whether to keep unnamed clusters when some character information is
            provided. If False, unnamed clusters will all be treated as noise.
            Defaults to True.
        accept_multiple_candidates (bool):
            Whether we try to perform classification when there are multiple
            candidate labels when classifying using metadata.
            Defaults to False.
        clu_min_samples (int):
            Minimum number of samples in a cluster. Defaults to 5.
        merge_threshold (float):
            Threshold for merging clusters. Defaults to 0.85.
        same_threshold_rel (float):
            The relative threshold for determining whether images belong to the same
            cluster for noise extraction and filtering. Defaults to 0.6.
        same_threshold_abs (int):
            The absolute threshold for determining whether images belong to the same
            cluster for noise extraction and filtering. Defaults to 10.
        n_add_images_to_ref (int):
            The number of images to add to the reference directory per character.
            Defaults to 0.
        move (bool):
            Whether to move or copy files
        logger (Optional[logging.Logger]):
            A logger to use for logging. Defaults to None, in which case
            the default logger will be used.
    """
    if logger is None:
        logger = logging.getLogger()
    (
        image_files,
        images,
        characters_per_image,
        character_mapping,
    ) = load_image_features_and_characters(
        src_dir, tqdm_desc="Extract dataset features", logger=logger
    )

    if ignore_character_metadata:
        characters_per_image = None
        character_mapping = dict()

    labels, batch_diff, batch_same = cluster_characters_basics(
        images,
        clu_min_samples=clu_min_samples,
        logger=logger,
    )

    # The number of known character names from metadata
    n_meta_labels = len(character_mapping)

    ref_images, ref_labels = None, None

    if ref_dir is not None:
        ref_image_files, ref_labels_tmp, ref_characters = parse_ref_dir(ref_dir)
        if ref_image_files:
            ref_images = load_image_features_and_characters(
                image_files=ref_image_files,
                tqdm_desc="Extract reference features",
                logger=logger,
            )[1]
            ref_labels = ref_labels_tmp
            # Merge class names and update ref_labels and characters_per_image
            character_mapping, ref_labels, characters_per_image = merge_characters(
                character_mapping, ref_characters, ref_labels, characters_per_image
            )

    # The number of known character names from either reference or metadata
    n_pre_labels = len(character_mapping)
    labels[labels >= 0] += n_pre_labels

    if to_extract_from_noise:
        extract_from_noise(
            image_files,
            images,
            labels=labels,
            batch_diff=batch_diff,
            batch_same=batch_same,
            characters_per_image=characters_per_image,
            ref_images=ref_images,
            ref_labels=ref_labels,
            same_threshold_rel=same_threshold_rel,
            same_threshold_abs=same_threshold_abs,
            logger=logger,
        )

    updated_indices_mapping = dict()

    if ref_images is not None:
        # Use a low threshold here because cluster may represent
        # different forms of the same character
        labels, updated_indices_mapping = map_clusters_to_reference(
            images,
            ref_images,
            ref_labels,
            cluster_ids=labels,
            same_threshold=0.01,
            characters_per_image=characters_per_image,
            logger=logger,
        )

    if characters_per_image is not None:
        labels, updated_indices_mapping_tmp = map_clusters_to_existing(
            labels,
            # Only retrieve the part that come from metadata
            characters_per_image[:, :n_meta_labels],
            n_pre_labels,
            min_proportion=0.6,
            accept_multiple_candidates=accept_multiple_candidates,
            logger=logger,
        )
        for meta_label in updated_indices_mapping_tmp:
            if meta_label in updated_indices_mapping:
                updated_indices_mapping[meta_label].extend(
                    updated_indices_mapping_tmp[meta_label]
                )
            else:
                updated_indices_mapping[meta_label] = updated_indices_mapping_tmp[
                    meta_label
                ]

    if ref_images is None and characters_per_image is None:
        keep_unnamed = True

    if keep_unnamed:
        # trying to merge clusters
        merge_clusters(
            labels,
            batch_same,
            min_merge_id=n_pre_labels,
            merge_threshold=merge_threshold,
            characters_per_image=characters_per_image,
            logger=logger,
        )
    else:
        labels[labels >= n_pre_labels] = -1

    if to_extract_from_noise:
        extract_from_noise(
            image_files,
            images,
            labels=labels,
            batch_diff=batch_diff,
            batch_same=batch_same,
            characters_per_image=characters_per_image,
            same_threshold_rel=same_threshold_rel,
            same_threshold_abs=same_threshold_abs,
            logger=logger,
        )

    if to_filter:
        labels = filter_characters_from_images(
            image_files,
            images,
            labels,
            batch_diff,
            batch_same,
            same_threshold_rel=same_threshold_rel,
            same_threshold_abs=same_threshold_abs,
            logger=logger,
        )

    if n_add_images_to_ref > 0:
        selected_indices, labels_for_ref = select_to_add_to_ref(
            updated_indices_mapping,
            n_add_images_to_ref,
        )
        image_files_for_ref = image_files[selected_indices]
        images_for_ref = images[selected_indices]
        # Note that the labels in updated_indices_mapping should all be part of
        # character_mapping as they come from either reference or metadata
        save_to_dir(
            image_files_for_ref,
            images_for_ref,
            ref_dir,
            labels_for_ref,
            character_mapping,
            logger=logger,
        )

    save_to_dir(
        image_files,
        images,
        dst_dir,
        labels,
        character_mapping,
        move=move,
        logger=logger,
    )
    remove_empty_folders(src_dir)
