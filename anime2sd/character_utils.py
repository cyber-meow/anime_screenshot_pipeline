import os
import json
import shutil
import random
import string
import logging
from tqdm import tqdm
from natsort import natsorted
from hbutils.string import plural_word

import numpy as np
from sklearn.cluster import OPTICS

from imgutils.metrics import ccip_extract_feature, ccip_default_threshold
from imgutils.metrics import ccip_difference, ccip_batch_differences
from anime2sd.basics import get_images_recursively
from anime2sd.basics import get_corr_meta_names, get_corr_ccip_names
from anime2sd.basics import get_default_metadata


def random_string(length=6):
    """Generate a random string of given length."""
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def remove_empty_folders(path_abs):
    """Remove empty folders recursively.

    Args:
        path_abs (str): The absolute path of the root folder.
    """
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def save_to_dir(image_files, images, dst_dir, labels, class_names=None, move=False):
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        if class_names and label != -1:
            folder_name = f"{int(label)}_{class_names[label]}"
        elif label == -1:
            folder_name = "-1_noise"
        else:
            folder_name = f"{int(label)}_{random_string()}"

        os.makedirs(os.path.join(dst_dir, folder_name), exist_ok=True)
        total = (labels == label).sum()
        logging.info(f'class {folder_name} has {plural_word(total, "image")} in total.')

        for img_path, img in zip(image_files[labels == label], images[labels == label]):
            img_path_dst = os.path.join(
                dst_dir, folder_name, os.path.basename(img_path)
            )
            if move:
                shutil.move(img_path, img_path_dst)
            else:
                shutil.copy(img_path, img_path_dst)

            # Handle metadata files
            meta_path, meta_filename = get_corr_meta_names(img_path)
            meta_path_dst = os.path.join(dst_dir, folder_name, meta_filename)

            if os.path.exists(meta_path):
                if move:
                    shutil.move(meta_path, meta_path_dst)
                else:
                    shutil.copyfile(meta_path, meta_path_dst)
            else:
                meta_data = get_default_metadata(img_path, warn=True)
                with open(meta_path_dst, "w") as meta_file:
                    json.dump(meta_data, meta_file, indent=4)

            ccip_path, ccip_filename = get_corr_ccip_names(img_path)
            ccip_path_dst = os.path.join(dst_dir, folder_name, ccip_filename)
            if os.path.exists(ccip_path):
                if move:
                    shutil.move(ccip_path, ccip_path_dst)
                else:
                    shutil.copy(ccip_path, ccip_path_dst)
            else:
                np.save(ccip_path_dst, img)


def parse_ref_dir(ref_dir):
    """
    Parse the reference directory to extract image files,
    labels, and class names.

    Args:
        ref_dir (str): Path to the reference directory.

    Returns:
        tuple: ref_image_files (list of image file paths),
              labels (list of labels corresponding to each image),
              class_names (dict mapping labels to class names).
    """
    ref_image_files = []
    labels = []
    class_names = {}
    label_counter = 0

    # Supported image extensions
    image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif"]

    # Check if there are class folders containing images
    subdirs = [
        d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d))
    ]
    if subdirs:
        for subdir in subdirs:
            class_name = subdir
            class_names[label_counter] = class_name
            for filename in os.listdir(os.path.join(ref_dir, subdir)):
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    ref_image_files.append(os.path.join(ref_dir, subdir, filename))
                    labels.append(label_counter)
            label_counter += 1
    else:
        # Images directly in the ref_dir
        for filename in os.listdir(ref_dir):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                class_name = (
                    filename.split("_")[0]
                    if "_" in filename
                    else os.path.splitext(filename)
                )[0]
                if class_name not in class_names.values():
                    class_names[label_counter] = class_name
                    current_label = label_counter
                    label_counter += 1
                else:
                    current_label = [
                        k for k, v in class_names.items() if v == class_name
                    ][0]
                ref_image_files.append(os.path.join(ref_dir, filename))
                labels.append(current_label)

    return ref_image_files, np.array(labels).astype(int), class_names


def classify_characters_map_clusters(
    imgs,
    ref_labels,
    cluster_ids,
    mode="min",
    same_threshold=0.9,
    ref_images=None,
    batch_diff=None,
    batch_same=None,
    desc=None,
):
    logging.info("Classifying ...")

    cls_labels = -1 * np.ones(len(imgs)).astype(int)
    max_label = np.max(ref_labels)

    unique_clusters = set(cluster_ids)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    for cluster_id in unique_clusters:
        cluster_indices = [
            idx for idx, img in enumerate(imgs) if cluster_ids[idx] == cluster_id
        ]

        if batch_diff is not None:
            avg_cluster_diff = np.mean(
                [batch_diff[idx] for idx in cluster_indices], axis=0
            )
        else:
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

        if batch_same is not None:
            r_sames = np.array(
                [
                    batch_same[ref_labels == label].mean()
                    for label in range(0, max_label + 1)
                ]
            )
        else:
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


def classify_characters(
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
    images_noise = images[labels == -1]
    noise_new_labels = classify_characters(
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
    images_noise = images[labels == -1]

    logging.info("Noise extracting complete.")
    label_cnt = {
        i: (labels == i).sum()
        for i in range(-1, max(labels) + 1)
        if (labels == i).sum() > 0
    }
    logging.info(f"Current label count: {label_cnt}")


def merge_clusters(exist_ids, max_clu_id, batch_same, labels, merge_threshold=0.85):
    # Perform in place operations to merge clusters
    while True:
        _round_merged = False
        for xi in range(0, max_clu_id + 1):
            if xi not in exist_ids:
                continue
            for yi in range(xi + 1, max_clu_id + 1):
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


def cluster_characters(
    images,
    merge_threshold: float = 0.85,
    clu_min_samples: int = 5,
    to_extract_from_noise: bool = True,
    to_merge_clusters: bool = True,
):
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

    if to_extract_from_noise:
        extract_from_noise(images, labels, batch_diff, batch_same)

    if to_merge_clusters:
        # trying to merge clusters
        _exist_ids = set(range(0, max_clu_id + 1))
        merge_clusters(_exist_ids, max_clu_id, batch_same, labels, merge_threshold=0.85)

    return labels, batch_diff, batch_same


def load_image_features(src_dir):
    image_files = np.array(natsorted(get_images_recursively(src_dir)))
    logging.info(f'Extracting feature of {plural_word(len(image_files), "images")} ...')
    images = []
    for img_path in tqdm(image_files, desc="Extract dataset features"):
        ccip_path, _ = get_corr_ccip_names(img_path)
        if os.path.exists(ccip_path):
            images.append(np.load(ccip_path))
        else:
            images.append(ccip_extract_feature(img_path))
    images = np.array(images)
    return image_files, images


def cluster_from_directory(
    src_dir,
    dst_dir,
    merge_threshold: float = 0.85,
    clu_min_samples: int = 5,
    to_extract_from_noise: bool = True,
    to_merge_clusters: bool = True,
    move: bool = False,
):
    image_files, images = load_image_features(src_dir)
    labels, _, _ = cluster_characters(
        images,
        merge_threshold,
        clu_min_samples,
        to_extract_from_noise,
        to_merge_clusters,
    )

    save_to_dir(image_files, images, dst_dir, labels, move=move)
    remove_empty_folders(src_dir)


def classify_from_directory(
    src_dir,
    dst_dir,
    ref_dir,
    clu_min_samples: int = 5,
    to_extract_from_noise: bool = True,
    move: bool = False,
):
    image_files, images = load_image_features(src_dir)

    clu_ids, batch_diff, batch_same = cluster_characters(
        images,
        clu_min_samples=clu_min_samples,
        to_merge_clusters=False,
        to_extract_from_noise=False,
    )

    ref_image_files, ref_labels, class_names = parse_ref_dir(ref_dir)
    logging.info(
        "Extracting feature of " + f'{plural_word(len(ref_image_files), "images")} ...'
    )
    ref_images = np.array(
        [
            ccip_extract_feature(img)
            for img in tqdm(ref_image_files, desc="Extract reference features")
        ]
    )
    clu_ids[clu_ids >= 0] += max(ref_labels) + 1
    if to_extract_from_noise:
        extract_from_noise(
            images,
            labels=clu_ids,
            batch_diff=batch_diff,
            batch_same=batch_same,
            ref_images=ref_images,
            ref_labels=ref_labels,
        )
    # Use a low threshold here because cluster may represent
    # different forms of the same character
    labels = classify_characters_map_clusters(
        images,
        ref_labels,
        same_threshold=0.1,
        cluster_ids=clu_ids,
        ref_images=ref_images,
        desc="Classifying images",
    )
    save_to_dir(image_files, images, dst_dir, labels, class_names, move=move)
    remove_empty_folders(src_dir)
