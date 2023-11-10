import logging
from tqdm import tqdm
from typing import Optional, Tuple, Dict
from hbutils.string import plural_word

import numpy as np
from sklearn.cluster import OPTICS

from imgutils.metrics import ccip_extract_feature, ccip_default_threshold
from imgutils.metrics import ccip_batch_differences

from anime2sd.basics import remove_empty_folders
from anime2sd.classif.imagewise import extract_from_noise, filter_characters_from_images
from anime2sd.classif.file_utils import load_image_features_and_characters
from anime2sd.classif.file_utils import parse_ref_dir, save_to_dir

from anime2sd.classif.merge_cluster import merge_clusters
from anime2sd.classif.merge_cluster import map_clusters_to_existing
from anime2sd.classif.merge_cluster import map_clusters_to_reference


def cluster_characters_basics(
    images: np.ndarray,
    clu_min_samples: int = 5,
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

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - labels (np.ndarray): An array of cluster labels assigned to each image.
        - batch_diff (np.ndarray):
            A square array of pairwise ccip differences between images.
        - batch_same (np.ndarray):
            A boolean array indicating if pairwise differences are below the threshold.

    """
    batch_diff = ccip_batch_differences(images)
    batch_same = batch_diff <= ccip_default_threshold()

    # clustering
    def _metric(x, y):
        return batch_diff[int(x), int(y)].item()

    logging.info("Clustering ...")
    samples = np.arange(len(images)).reshape(-1, 1)
    # max_eps, _ = ccip_default_clustering_params(method='optics_best')
    clustering = OPTICS(min_samples=clu_min_samples, metric=_metric).fit(samples)
    labels = clustering.labels_.astype(int)

    max_clu_id = labels.max().item()
    all_label_ids = np.array([-1, *range(0, max_clu_id + 1)])
    logging.info(f'Cluster complete, with {plural_word(max_clu_id, "cluster")}.')
    label_cnt = {
        i: (labels == i).sum() for i in all_label_ids if (labels == i).sum() > 0
    }
    logging.info(f"Current label count: {label_cnt}")

    return labels, batch_diff, batch_same


def merge_class_names(
    class_names: Dict[int, str],
    ref_class_names: Dict[int, str],
    ref_labels: np.ndarray,
    characters_per_image: Optional[np.ndarray],
) -> Tuple[Dict[int, str], np.ndarray, Optional[np.ndarray]]:
    """
    Merge reference class names into existing class names, update labels,
    and adjust characters_per_image.

    Args:
        class_names (Dict[int, str]):
            Existing class names with labels as keys.
        ref_class_names (Dict[int, str]):
            Reference class names with labels as keys.
        ref_labels (np.ndarray):
            Array of labels corresponding to reference images.
        characters_per_image (Optional[np.ndarray]):
            Array indicating presence of characters in images.

    Returns:
        Tuple[Dict[int, str], np.ndarray, Optional[np.ndarray]]:
            - Updated class names dictionary.
            - Updated reference labels.
            - Updated characters_per_image if provided.
    """
    # Create a new class_names dictionary to avoid modifying the original one
    updated_class_names = class_names.copy()
    updated_ref_labels = ref_labels.copy()
    n_existing_labels = max(class_names.keys()) + 1 if class_names else 0

    # Update the class_names with ref_class_names
    for ref_label, ref_name in ref_class_names.items():
        existing_label = next(
            (label for label, name in class_names.items() if name == ref_name), None
        )
        if existing_label is not None:
            # Update ref_labels with the existing label from class_names
            updated_ref_labels[ref_labels == ref_label] = existing_label
        else:
            # Add new entry to class_names if it doesn't exist
            updated_class_names[n_existing_labels] = ref_name
            updated_ref_labels[ref_labels == ref_label] = n_existing_labels
            n_existing_labels += 1

    # Update characters_per_image for new labels based on the first word of the
    # class name
    if characters_per_image is not None:
        new_characters_per_image = np.zeros(
            (characters_per_image.shape[0], n_existing_labels), dtype=bool
        )
        new_characters_per_image[
            :, : characters_per_image.shape[1]
        ] = characters_per_image

        for new_label in range(characters_per_image.shape[1], n_existing_labels):
            new_class_name = updated_class_names[new_label]

            # Find an existing label whose name matches the new class name exactly or
            # is a prefix of it
            old_label = next(
                (
                    label
                    for label, name in class_names.items()
                    if new_class_name == name
                    or new_class_name.startswith(name + " ")
                    or new_class_name.startswith(name + "_")
                ),
                None,
            )
            if old_label is not None:
                # Copy the character presence data from the old label column to
                # the new label column
                new_characters_per_image[:, new_label] = new_characters_per_image[
                    :, old_label
                ]

        characters_per_image = new_characters_per_image

    return updated_class_names, updated_ref_labels, characters_per_image


def classify_from_directory(
    src_dir: str,
    dst_dir: str,
    ref_dir: Optional[str] = None,
    ignore_character_metadata: bool = False,
    to_extract_from_noise: bool = True,
    to_filter: bool = True,
    keep_unnamed: bool = True,
    clu_min_samples: int = 5,
    merge_threshold: float = 0.85,
    same_threshold_rel: float = 0.6,
    same_threshold_abs: int = 10,
    move: bool = False,
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
        * Merging clusters based on similarity
          (if no reference images or metadata is available).
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
        move: Whether to move or copy files
    """
    (
        image_files,
        images,
        characters_per_image,
        class_names,
    ) = load_image_features_and_characters(src_dir)

    if ignore_character_metadata:
        characters_per_image = None
        class_names = dict()

    labels, batch_diff, batch_same = cluster_characters_basics(
        images,
        clu_min_samples=clu_min_samples,
    )

    # The number of known character names from metadata
    n_meta_labels = len(class_names)

    if ref_dir is not None:
        ref_image_files, ref_labels, ref_class_names = parse_ref_dir(ref_dir)
        logging.info(
            "Extracting feature of "
            + f'{plural_word(len(ref_image_files), "images")} ...'
        )
        ref_images = np.array(
            [
                ccip_extract_feature(img)
                for img in tqdm(ref_image_files, desc="Extract reference features")
            ]
        )
        # Merge class names and update ref_labels and characters_per_image
        class_names, ref_labels, characters_per_image = merge_class_names(
            class_names, ref_class_names, ref_labels, characters_per_image
        )
    else:
        ref_images, ref_labels = None, None

    # The number of known character names from either reference or metadata
    n_pre_labels = len(class_names)
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
        )

    if ref_images is not None:
        # Use a low threshold here because cluster may represent
        # different forms of the same character
        labels = map_clusters_to_reference(
            images,
            ref_images,
            ref_labels,
            cluster_ids=labels,
            same_threshold=0.1,
            characters_per_image=characters_per_image,
        )

    if characters_per_image is not None:
        labels = map_clusters_to_existing(
            labels,
            characters_per_image[:, :n_meta_labels],
            n_pre_labels,
            min_proportion=0.6,
        )

    if ref_images is None and characters_per_image is None:
        keep_unnamed = True

    if keep_unnamed:
        # trying to merge clusters
        _exist_ids = np.unique(labels[labels >= 0])
        max_clu_id = np.max(_exist_ids)
        merge_clusters(
            set(_exist_ids),
            max_clu_id,
            batch_same,
            labels,
            min_merge_id=n_pre_labels,
            merge_threshold=merge_threshold,
            characters_per_image=characters_per_image,
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
        )

    save_to_dir(image_files, images, dst_dir, labels, class_names, move=move)
    remove_empty_folders(src_dir)
