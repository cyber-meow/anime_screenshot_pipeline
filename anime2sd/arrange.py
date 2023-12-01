import os
import re
import json
import shutil
import logging
from tqdm import tqdm

from anime2sd.basics import get_images_recursively, get_files_recursively
from anime2sd.basics import get_corr_meta_names
from anime2sd.character import Character


def construct_aux_files_dict(paths):
    """
    Construct a dictionary where keys are the img_base values
    and the values are lists of files that are auxiliary to that img_base.

    :param files: List of all files.
    :return: Dictionary with img_base as keys and
        lists of auxiliary files as values.
    """
    aux_dict = {}

    for path in tqdm(paths):
        dirname, filename = os.path.split(path)
        if filename.startswith("."):
            file_base = re.sub(r"\_meta.json$", "", filename).lstrip(".")
        else:
            file_base = os.path.splitext(filename)[0]
        file_base = os.path.join(dirname, file_base)
        if file_base not in aux_dict:
            aux_dict[file_base] = [path]
        else:
            aux_dict[file_base].append(path)
    return aux_dict


def move_img_with_aux(img_path, dst_dir, aux_dict):
    """
    Move image along with its auxiliary files to a destination directory.

    :param img_path: Path to the image file.
    :param dst_dir: Destination directory.
    :return: None
    """
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    img_base, _ = os.path.splitext(img_path)
    for path in aux_dict[img_base]:
        if os.path.dirname(path) != dst_dir:
            shutil.move(path, dst_dir)


def get_folder_name(folder_type, info_dict, max_character_number) -> str:
    valid_folder_types = [
        "n_people",
        "n_characters",
        "character",
    ]
    assert folder_type in valid_folder_types, f"invalid folder type {folder_type}"
    if folder_type == "n_people":
        count = info_dict["n_people"]
        suffix = "person" if count == 1 else "people"
        return f"{count}_{suffix}"
    elif folder_type == "n_characters":
        characters = info_dict.get("characters", [])
        characters = [
            Character.from_string(character).character_name
            for character in info_dict.get("characters", [])
        ]
        characters = sorted(list(set(characters)))
        n_character = len(characters)
        if n_character >= max_character_number:
            return f"{max_character_number}+_characters"
        suffix = "character" if n_character == 1 else "characters"
        return f"{n_character}_{suffix}"
    elif folder_type == "character":
        characters = [
            Character.from_string(character).character_name
            for character in info_dict.get("characters", [])
        ]
        characters = sorted(list(set(characters)))
        if len(characters) == 0:
            return "character_others"
        character_folder = "+".join(sorted(characters))
        # Cannot have folder of too long name
        if len(character_folder) >= 100:
            return "character_others"
        else:
            return character_folder.replace(".", "")
    # This cannot happen
    else:
        exit(1)


def get_dst_dir(info_dict, dst_dir, arrange_format, max_character_number):
    if arrange_format == "":
        return dst_dir, None

    folder_types = arrange_format.split("/")
    character_folder = None
    for folder_type in folder_types:
        folder_name = get_folder_name(folder_type, info_dict, max_character_number)
        dst_dir = os.path.join(dst_dir, folder_name)
        if folder_type == "character":
            character_folder = folder_name

    return dst_dir, character_folder


def count_n_images(filenames):
    count = 0
    # Iterate through the list of filenames
    for filename in filenames:
        # Get the file extension
        extension = os.path.splitext(filename)[1]
        # Check if the extension is one of the common image file extensions
        if extension.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            # If it is, increment the count
            count += 1
    return count


def merge_folder(character_comb_dict, min_images_per_comb, aux_dict, logger):
    if logger is None:
        logger = logging.getLogger()
    for comb in tqdm(character_comb_dict):
        files = character_comb_dict[comb]
        n_images = count_n_images(files)
        if n_images < min_images_per_comb:
            logger.info(
                f"{comb} has fewer than {min_images_per_comb} images; "
                + "renamed as character_others"
            )
            for file in files:
                new_path = file.replace(comb, "character_others")
                dst_dir = os.path.dirname(new_path)
                move_img_with_aux(file, dst_dir, aux_dict)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def arrange_folder(
    src_dir,
    dst_dir,
    arrange_format,
    max_character_number,
    min_images_per_combination,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger()
    img_paths = get_images_recursively(src_dir)
    file_paths = get_files_recursively(src_dir)

    logger.info("Constructing dictionary for auxiliary files...")
    aux_dict = construct_aux_files_dict(file_paths)
    character_combination_dict = dict()

    logger.info("Rearranging...")
    for img_path in tqdm(img_paths):
        meta_file_path, _ = get_corr_meta_names(img_path)

        if os.path.exists(meta_file_path):
            with open(meta_file_path, "r") as meta_file:
                meta_data = json.load(meta_file)
        else:
            raise ValueError(f"Metadata unfound for {img_path}")

        img_dst_dir, character_folder = get_dst_dir(
            meta_data, dst_dir, arrange_format, max_character_number
        )

        move_img_with_aux(img_path, img_dst_dir, aux_dict)

        new_path = os.path.join(img_dst_dir, os.path.basename(img_path))
        if character_folder is not None:
            if character_folder in character_combination_dict:
                character_combination_dict[character_folder].append(new_path)
            else:
                character_combination_dict[character_folder] = [new_path]

    # We need to cosntruct the dictionary again after moving the files
    logger.info("Constructing dictionary for auxiliary files...")
    file_paths = get_files_recursively(src_dir)
    aux_dict = construct_aux_files_dict(file_paths)
    logger.info("Merging folders...")
    if min_images_per_combination > 1:
        merge_folder(
            character_combination_dict,
            min_images_per_combination,
            aux_dict,
            logger=logger,
        )
    remove_empty_folders(src_dir)
    remove_empty_folders(dst_dir)
