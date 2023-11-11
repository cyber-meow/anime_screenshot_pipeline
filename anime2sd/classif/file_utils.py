import os
import json
import shutil
import logging
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List
from natsort import natsorted
from hbutils.string import plural_word

import numpy as np

from imgutils.metrics import ccip_extract_feature
from anime2sd.basics import random_string
from anime2sd.basics import get_images_recursively
from anime2sd.basics import get_corr_meta_names, get_corr_ccip_names
from anime2sd.basics import get_default_metadata


def load_image_features_and_characters(
    src_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[int, str]]:
    """Load image features and associated character information
    from a given source directory.

    Args:
        src_dir (str): The source directory where image files and
                       their corresponding metadata are stored.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
            - A numpy array of image file paths.
            - A numpy array of extracted image features.
            - A boolean array indicating the presence of characters in each image.
              None if no characters are found.
            - A dictionary mapping label indices to class names.
    """
    image_files = np.array(natsorted(get_images_recursively(src_dir)))
    logging.info(f'Extracting feature of {plural_word(len(image_files), "image")} ...')
    images = []
    characters_list = []
    character_to_index = {}  # Mapping from character name to label index
    index_to_character = {}  # Mapping from label index to character name
    label_counter = 0

    # Iterate over image files to extract features and metadata
    for img_path in tqdm(image_files, desc="Extract dataset features"):
        ccip_path, _ = get_corr_ccip_names(img_path)
        if os.path.exists(ccip_path):
            images.append(np.load(ccip_path))
        else:
            images.append(ccip_extract_feature(img_path))

        meta_path, _ = get_corr_meta_names(img_path)
        characters = []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as meta_file:
                meta_data = json.load(meta_file)
                if "characters" in meta_data:
                    for character in meta_data["characters"]:
                        if character not in character_to_index:
                            character_to_index[character] = label_counter
                            index_to_character[label_counter] = character
                            label_counter += 1
                        characters.append(character_to_index[character])
        characters_list.append(characters)

    images = np.array(images)

    # Initialize characters_per_image only if there are characters found
    characters_per_image = None
    if character_to_index:
        characters_per_image = np.full(
            (len(images), len(character_to_index)), False, dtype=bool
        )
        # Set True where character is present
        for i, character_indices in enumerate(characters_list):
            for character_index in character_indices:
                characters_per_image[i, character_index] = True
    else:
        logging.info(
            "No character metadata found; returning None for 'characters_per_image'."
        )

    return image_files, images, characters_per_image, index_to_character


def parse_ref_dir(ref_dir: str) -> Tuple[List[str], np.ndarray, Dict[int, str]]:
    """
    Parse the reference directory to extract image files, their labels, and class names.
    This function assumes that either
        - The directory contains subdirectories named after their class names,
          each containing relevant images.
        - The directory contains class names from the image file names,
          expecting an underscore '_' delimiter.

    Args:
        ref_dir (str): Path to the reference directory.

    Returns:
        Tuple[List[str], np.ndarray, Dict[int, str]]:
            - A list of paths to image files.
            - An array of integer labels corresponding to each image,
              where each label is an index in class_names.
            - A dictionary mapping label indices to class names.
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


def save_to_dir(
    image_files: List[str],
    images: np.ndarray,
    dst_dir: str,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    move: bool = False,
) -> None:
    """
    Save or move image files to a destination directory, organized into
    subdirectories by label.

    Args:
        image_files (List[str]):
            A list of paths to image files.
        images (np.ndarray):
            An array of extracted image features.
        dst_dir (str):
            The destination directory where image files will be saved or moved.
        labels (np.ndarray):
            An array of integer labels corresponding to each image.
        class_names (Optional[Dict[int, str]]):
            A dictionary mapping integer labels to class names.
            Defaults to None, in which case random strings will be used.
        move (bool):
            If True, move files instead of copying. Defaults to False.
    """
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        if class_names and label in class_names:
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
