import os
import json
import random
import logging
import shutil
from tqdm import tqdm
from typing import List
from PIL import Image

import fiftyone.zoo as foz
from waifuc.action import ThreeStageSplitAction

from .basics import (
    get_images_recursively,
    get_folders_recursively,
    get_corr_meta_names,
    get_or_generate_metadata,
    remove_empty_folders,
)
from .emb_utils import update_emb_init_info
from .extract_and_remove_similar import remove_similar_from_dir
from .character import Character
from .waifuc_customize import LocalSource, SaveExporter


def parse_char_name(folder_name: str) -> str:
    """parse the character name from the folder name

    Args:
        folder_name (str): The name of the folder

    Returns:
        str: The character name
    """
    if "_" in folder_name:
        parts = folder_name.split("_")
        if parts[0].strip("-").isdigit():
            return "_".join(parts[1:])
    return folder_name


def save_characters_to_meta(
    classified_dir: str,
    overwrite_uncropped: bool = True,
    overwrite_path: bool = False,
) -> List[str]:
    """Save character information to metadata files

    Args:
        classified_dir (str):
            Directory containing classified character folders
        overwrite_raw (bool):
            Whether to overwrite the character metadata of uncropped images or not
        overwrite_path (bool):
            Whether to overwrite the path of the character metadata

    Returns:
        List[str]: List of character embedding names
    """

    # To keep track of paths encountered in this run
    # Value set to True if the character list is to be updated
    encountered_paths = dict()
    characters = set()

    logging.info("Saving characters to metadata ...")
    # Iterate over each folder in the classified directory
    for folder_name in tqdm(get_folders_recursively(classified_dir)):
        folder_name = os.path.relpath(folder_name, classified_dir)
        char_name = parse_char_name(folder_name)
        character = Character.from_string(char_name, outer_sep=os.path.sep)

        if not char_name.lower().startswith("noise"):
            characters.add(character.embedding_name)
        folder_path = os.path.join(classified_dir, folder_name)

        # Iterate over each image file in the folder
        for img_file in os.listdir(folder_path):
            img_name, img_ext = os.path.splitext(img_file)

            # Ensure it's an image file
            if img_ext.lower() not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                continue

            img_path = os.path.join(folder_path, img_file)
            meta_file_path, _ = get_corr_meta_names(img_path)
            meta_data = get_or_generate_metadata(
                img_path, warn=True, overwrite_path=overwrite_path
            )

            # Update the characters field
            if char_name.startswith("noise") or char_name.startswith("Noise"):
                # This ensures that we overwrite old information
                meta_data["characters"] = []
            else:
                meta_data["characters"] = [character.to_string()]

            # Save the updated metadata for the cropped image
            with open(meta_file_path, "w") as meta_file:
                json.dump(meta_data, meta_file, indent=4)

            # Check for the 'path' field and update it
            if "path" in meta_data:
                original_path = meta_data["path"]
                if original_path == img_path:
                    continue

                original_meta_path, _ = get_corr_meta_names(original_path)
                # Note that if we are here overwrite_path must be False
                orig_meta_data = get_or_generate_metadata(original_path, warn=False)

                # Initialize characters list if the path hasn't been encountered
                # yet in this run and overwrite_uncropped is True
                if original_path not in encountered_paths.keys():
                    if "characters" not in orig_meta_data or overwrite_uncropped:
                        orig_meta_data["characters"] = []
                        encountered_paths[original_path] = True
                    else:
                        encountered_paths[original_path] = False
                # Go to next image if we do not update
                if not encountered_paths[original_path]:
                    continue

                # Append the character name if it's not already in the list
                # and is not noise
                if character not in orig_meta_data[
                    "characters"
                ] and not char_name.lower().startswith("noise"):
                    orig_meta_data["characters"].append(character.to_string())

                # Save the updated original metadata
                with open(original_meta_path, "w") as orig_meta_file:
                    json.dump(orig_meta_data, orig_meta_file, indent=4)
    return list(characters)


def resize_image(image: Image, max_size: int) -> Image:
    """
    Resize the image to ensure its dimensions do not exceed the specified max_size.

    Args:
        image (Image): The PIL Image object to be resized.
        max_size (int): The maximum size for the smaller dimension of the image.

    Returns:
        Image: The resized image if resizing is needed; otherwise, the original image.
    """
    width, height = image.size
    if max_size > min(height, width):
        return image

    # Calculate the scaling factor
    scaling_factor = max_size / min(height, width)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def save_image_and_meta(
    img: Image, img_path: str, save_dir: str, ext: str, image_type: str
) -> None:
    """
    Save the image with the specified extension and
    update and copy its corresponding metadata.

    Args:
        img (Image): The PIL Image object to be saved.
        img_path (str): The path of the original image.
        save_dir (str): The directory where the image will be saved.
        ext (str): The extension to be used for saving the image.
        image_type (str): The type of the image to be recorded in the metadata.

    Raises:
        IOError: If there is an error saving the image.
        ValueError: If metadata file does not exist.
    """
    # Extract the filename from the original image path
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]

    # Adjust the filename to match the provided extension
    adjusted_filename = base_filename + ext
    adjusted_path = os.path.join(save_dir, adjusted_filename)

    try:
        if ext == ".webp":
            img.save(adjusted_path, format="WEBP", quality=95)
        else:
            img.save(adjusted_path)
    except IOError as e:
        logging.error(f"Error saving the image {adjusted_path}: {e}")

    # Copy the corresponding metadata file
    meta_path, meta_filename = get_corr_meta_names(img_path)
    meta_data = get_or_generate_metadata(img_path, warn=True)

    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
        _, ext_orig = os.path.splitext(meta_data["filename"])
        new_meta_path = os.path.join(save_dir, meta_filename)
        meta_data["current_path"] = new_meta_path
        meta_data["type"] = image_type
        meta_data["filename"] = meta_data["filename"].replace(ext_orig, ext)
        meta_data["image_size"] = img.size

        # Save the updated metadata with new extension
        with open(new_meta_path, "w") as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    # Normally this never gets triggered
    else:
        raise ValueError("All metadata must exist before resizing to dst")


def copy_image_and_meta(img_path: str, save_dir: str, image_type: str) -> None:
    """
    Copy the image and its corresponding metadata to a new directory.

    Args:
        img_path (str): The path of the original image.
        save_dir (str): The directory where the image will be copied.
        image_type (str): The type of the image to be recorded in the updated metadata.

    Raises:
        ValueError: If metadata file does not exist.
    """
    shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))
    # Copy the corresponding metadata file
    meta_path, meta_filename = get_corr_meta_names(img_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
        new_meta_path = os.path.join(save_dir, meta_filename)
        meta_data["current_path"] = new_meta_path
        meta_data["type"] = image_type

        # Save the updated metadata with new extension
        with open(new_meta_path, "w") as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    # Normally this never gets triggered
    else:
        raise ValueError("All metadata must exist before resizing to dst")


def resize_character_images(
    src_dirs: List[str],
    dst_dir: str,
    max_size: int,
    ext: str,
    image_type: str,
    n_nocharacter_frames: int,
    to_resize: bool = True,
    overwrite_path: bool = False,
) -> None:
    """
    Process images from source directories, resize them if needed,
    and save them to the destination directory.

    Args:
        src_dirs (list):
            A list of source directories containing images.
        dst_dir (str):
            The destination directory to save processed images.
        max_size (int):
            The maximum size for the smaller dimension of the image.
        ext (str):
            The extension to be used for saving the images.
        image_type (str):
            The type of the images to be recorded in metadata.
        n_nocharacter_frames (int):
            The number of frames without characters to save.
        to_resize (bool, optional):
            Flag to determine if resizing is required. Defaults to True.
        overwrite_path (bool, optional):
            Flag to determine if existing paths should be overwritten.
            Defaults to False.

    Raises:
        ValueError: If there is an error reading an image.
        IOError: If there is an error processing an image.
    """
    nocharacter_frames = []
    processed_img_paths = set()
    for src_dir in src_dirs:
        if os.path.basename(src_dir) == "raw":
            warn = False
        else:
            warn = True
        logging.info(f"Processing images from {src_dir} ...")
        save_dir = os.path.join(dst_dir, os.path.basename(src_dir))
        os.makedirs(save_dir, exist_ok=True)

        for img_path in tqdm(get_images_recursively(src_dir)):
            if img_path in processed_img_paths:
                continue
            processed_img_paths.add(img_path)
            meta_data = get_or_generate_metadata(
                img_path, warn=warn, overwrite_path=overwrite_path
            )
            if "characters" in meta_data and meta_data["characters"]:
                original_path = meta_data["path"]
                if os.path.basename(src_dir) != "raw" and original_path != img_path:
                    orig_meta_data = get_or_generate_metadata(
                        original_path, warn=False, overwrite_path=overwrite_path
                    )
                    cropped_size = meta_data["image_size"]
                    cropped_area = cropped_size[0] * cropped_size[1]
                    orig_size = orig_meta_data["image_size"]
                    orig_area = orig_size[0] * orig_size[1]

                    if cropped_area > 0.5 * orig_area:
                        if original_path in processed_img_paths:
                            continue
                        processed_img_paths.add(original_path)
                        img_path = original_path

                if to_resize:
                    try:
                        img = Image.open(img_path)
                    except IOError:
                        raise ValueError(f"Error reading image: {img_path}")
                    resized_img = resize_image(img, max_size)
                    save_image_and_meta(
                        resized_img, img_path, save_dir, ext, image_type
                    )
                else:
                    copy_image_and_meta(img_path, save_dir, image_type)
            else:
                nocharacter_frames.append(img_path)

    # Process no character images
    # Randomly select n_nocharacter_frames and save
    save_dir = os.path.join(dst_dir, "no_characters")
    os.makedirs(save_dir, exist_ok=True)

    if n_nocharacter_frames < len(nocharacter_frames):
        selected_frames = random.sample(nocharacter_frames, n_nocharacter_frames)
    else:
        selected_frames = nocharacter_frames

    logging.info(f"Copying {len(selected_frames)} no character images ...")

    for img_path in tqdm(selected_frames):
        if to_resize:
            try:
                img = Image.open(img_path)
            except IOError:
                raise ValueError(f"Error reading image: {img_path}")
            resized_img = resize_image(img, max_size)
            save_image_and_meta(resized_img, img_path, save_dir, ext, image_type)
        else:
            copy_image_and_meta(img_path, save_dir, image_type)


def select_dataset_images_from_directory(
    classified_dir: str,
    full_dir: str,
    dst_dir: str,
    pipeline_type: str,
    overwrite_path: bool,
    # For saving character to metadata
    character_overwrite_uncropped: bool,
    # For saving embedding initialization information
    image_type: str,
    overwrite_emb_init_info: bool,
    # For 3 stage cropping
    use_3stage_crop: bool,
    detect_level: str,
    # For resizing/copying images to destination
    max_size: int,
    image_save_ext: str,
    to_resize: bool,
    n_anime_reg: int,
    # For additional filtering after obtaining dataset images
    filter_again: bool,
    detect_duplicate_model: str,
    similarity_threshold: float,
) -> None:
    """
    Select and process images from the specified directories for dataset creation.

    This function handles various operations like cropping, resizing, and filtering of
    images to prepare them for a dataset.

    Args:
        classified_dir (str): Directory containing classified images.
        full_dir (str): Directory containing the original images.
        dst_dir (str): Destination directory for processed images.
        pipeline_type (str): Type of the processing pipeline.
        character_overwrite_uncropped (bool):
            Flag to overwrite character metadata of uncropped images.
        image_type (str): Type of the images for metadata and embedding initialization.
        overwrite_emb_init_info (bool): Flag to overwrite embedding initialization info.
        use_3stage_crop (bool): Flag to use three-stage cropping.
        detect_level (str): Detection level for cropping.
        overwrite_path (bool): Flag to overwrite existing paths in metadata.
        max_size (int): Maximum size for image resizing.
        image_save_ext (str): Extension for saving images.
        to_resize (bool): Flag to resize images.
        n_anime_reg (int): Number of no character anime images to save.
        filter_again (bool): Flag to filter similar images again after processing.
        detect_duplicate_model (str): Model used for duplicate detection.
        similarity_threshold (float): Threshold for similarity in duplicate detection.

    Raises:
        ValueError: If there is an error in processing the images.
        IOError: If there is an error in reading or writing images.
    """
    # [Function body]

    overwrite_uncropped = (
        pipeline_type == "screenshots" or character_overwrite_uncropped
    )
    # update metadata using folder name
    character_embeddings = save_characters_to_meta(
        classified_dir, overwrite_uncropped, overwrite_path=overwrite_path
    )

    # save trigger word info
    emb_init_filepath = os.path.join(dst_dir, "emb_init.json")
    update_emb_init_info(
        emb_init_filepath,
        character_embeddings,
        image_type,
        overwrite=overwrite_emb_init_info,
    )

    if use_3stage_crop:
        if detect_level in ["s", "n"]:
            detect_level_head_halfbody = detect_level
        else:
            # Use the faster model by default
            detect_level_head_halfbody = "n"
        detect_config = {"level": detect_level_head_halfbody}
        crop_action = ThreeStageSplitAction(
            split_person=False,
            head_conf=detect_config,
            halfbody_conf=detect_config,
        )
        logging.info(f"Performing 3 stage cropping for {classified_dir} ...")
        source = LocalSource(classified_dir)
        source.attach(
            crop_action,
        ).export(SaveExporter(classified_dir, in_place=True))

    n_reg = n_anime_reg if pipeline_type == "screenshots" else 0
    # select images, resize, and save to training
    resize_character_images(
        [classified_dir, full_dir],
        dst_dir,
        max_size=max_size,
        ext=image_save_ext,
        image_type=image_type,
        n_nocharacter_frames=n_reg,
        to_resize=to_resize,
        overwrite_path=overwrite_path,
    )
    remove_empty_folders(dst_dir)

    if filter_again:
        logging.info(f"Removing duplicates from {dst_dir} ...")
        model = foz.load_zoo_model(detect_duplicate_model)
        for folder in os.listdir(dst_dir):
            if os.path.isdir(os.path.join(dst_dir, folder)):
                remove_similar_from_dir(
                    os.path.join(dst_dir, folder),
                    model=model,
                    thresh=similarity_threshold,
                )
