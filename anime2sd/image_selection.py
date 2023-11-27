import os
import json
import random
import logging
import shutil
from tqdm import tqdm
from typing import List
from PIL import Image

from anime2sd.basics import get_images_recursively, get_folders_recursively
from anime2sd.basics import get_corr_meta_names, get_or_generate_metadata
from anime2sd.character import Character


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
    classified_dir: str, overwrite_uncropped: bool = True
) -> List[str]:
    """Save character information to metadata files

    Args:
        classified_dir (str):
            Directory containing classified character folders
        overwrite_raw (bool):
            Whether to overwrite the character metadata of uncropped images or not

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
            meta_data = get_or_generate_metadata(img_path, warn=True)

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


def resize_image(image, max_size):
    width, height = image.size
    if max_size > min(height, width):
        return image

    # Calculate the scaling factor
    scaling_factor = max_size / min(height, width)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def save_image_and_meta(img, img_path, save_dir, ext, image_type):
    """
    Save the image based on the provided extension.
    Adjusts the path to match the extension if necessary.
    Also copies the corresponding metadata file to the save directory.
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
        meta_data["filename"] = meta_data["filename"].replace(ext_orig, ext)
        meta_data["type"] = image_type
        meta_data["image_size"] = img.size

        # Save the updated metadata with new extension
        with open(os.path.join(save_dir, meta_filename), "w") as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    # Normally this never gets triggered
    else:
        raise ValueError("All metadata must exist before resizing to dst")


def copy_image_and_meta(img_path, save_dir, image_type):
    shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))
    # Copy the corresponding metadata file
    meta_path, meta_filename = get_corr_meta_names(img_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
        meta_data["type"] = image_type

        # Save the updated metadata with new extension
        with open(os.path.join(save_dir, meta_filename), "w") as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    # Normally this never gets triggered
    else:
        raise ValueError("All metadata must exist before resizing to dst")


def resize_character_images(
    src_dirs, dst_dir, max_size, ext, image_type, n_nocharacter_frames, to_resize=True
):
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
            meta_data = get_or_generate_metadata(img_path, warn=warn)
            if "characters" in meta_data and meta_data["characters"]:
                original_path = meta_data["path"]
                if os.path.basename(src_dir) != "raw" and original_path != img_path:
                    orig_meta_data = get_or_generate_metadata(original_path, warn=False)
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
