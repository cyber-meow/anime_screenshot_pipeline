import os
import re
import csv
import json
import random
import string

from PIL import Image
from typing import List
from pathlib import Path


def parse_anime_info(filename: str) -> tuple:
    """
    Parses a filename to extract the anime name and episode number.

    Args:
        filename (str): The filename to be parsed.

    Returns:
        tuple: A tuple containing the anime name and episode number.
    """
    # Remove square bracket contents
    filename = re.sub(r"\[.*?\]", "", filename)
    filename = os.path.splitext(filename)[0].strip()

    # Split on the last occurrence of '-'
    parts = filename.rsplit("-", 1)

    if len(parts) == 2:
        anime_name = parts[0].strip()
        episode_part = parts[1]

        # Extract episode number
        episode_num_match = re.search(r"^\W*\d+", episode_part)
        episode_num = int(episode_num_match.group(0)) if episode_num_match else None

        return anime_name, episode_num

    return filename, None


def to_list(line):
    items = [item.strip() for item in line.split(",")]
    items = [item for item in items if item not in ["", "unknown", "anonymous"]]
    return items


def parse_grabber_info(grabber_info: List[str]) -> dict:
    """
    Parses grabber information that follows the following format:

    character: %character:spaces,separator=^, %
    copyright: %copyright:spaces,separator=^, %
    artist: %artist:spaces,separator=^, %
    general: %general:spaces,separator=^, %
    rating: %rating%
    score: %score%

    Args:
        grabber_info (List[str]): The grabber information to be parsed.

    Returns:
        dict: A dictionary containing the parsed grabber information.
    """
    basic_info = {}
    for line in grabber_info:
        for field in ["characters", "copyright", "artist", "tags"]:
            if line.startswith(f"{field}: "):
                basic_info[field] = to_list(line.lstrip(f"{field}:"))
        for field in ["rating", "score", "fav_count"]:
            if line.startswith(f"{field}: "):
                basic_info[field] = line.lstrip(f"{field}:").strip()
        if line.startswith("character: "):
            basic_info["characters"] = to_list(line.lstrip("character:"))
        if line.startswith("general: "):
            basic_info["tags"] = to_list(line.lstrip("general:"))
    return basic_info


def random_string(length=6):
    """Generate a random string of given length."""
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def sanitize_path_component(component: str) -> str:
    """
    Sanitizes an individual component of a file path to be compatible
    with Windows file system.

    Args:
        component (str): The path component to be sanitized.

    Returns:
        str: A sanitized version of the path component.
    """
    # replace invalid characters with an underscore
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, "_", component)


def sanitize_path(path: str) -> str:
    """
    Sanitizes a file path by sanitizing each component of the path,
    except for the drive letter on Windows.

    Args:
        path (str): The original file path.

    Returns:
        str: A sanitized version of the file path.
    """
    # Split the path into components
    components = path.split(os.path.sep)

    # Special handling for Windows drive letter
    if len(components) > 1 and re.match(r"^[a-zA-Z]:$", components[0]):
        drive = components.pop(0)
        sanitized_components = [drive] + [
            sanitize_path_component(comp) for comp in components
        ]
    else:
        sanitized_components = [sanitize_path_component(comp) for comp in components]

    # Reassemble the sanitized path
    return os.path.sep.join(sanitized_components)


def remove_empty_folders(path_abs):
    """Remove empty folders recursively.

    Args:
        path_abs (str): The absolute path of the root folder.
    """
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def get_images_recursively(folder_path):
    """
    Get all images recursively from a folder using Path and rglob.

    Args:
    - folder_path (str): The path to the folder.

    Returns:
    - list: A list of image paths.
    """
    allowed_patterns = [
        "*.[Pp][Nn][Gg]",
        "*.[Jj][Pp][Gg]",
        "*.[Jj][Pp][Ee][Gg]",
        "*.[Ww][Ee][Bb][Pp]",
        "*.[Gg][Ii][Ff]",
    ]

    image_path_list = [
        str(path)
        for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def get_files_recursively(folder_path):
    """
    Get all files recursively from a folder using Path and rglob.

    Args:
    - folder_path (str): The path to the folder.

    Returns:
    - list: A list of file paths.
    """
    return [str(file) for file in Path(folder_path).rglob("*") if file.is_file()]


def get_folders_recursively(folder_path):
    """
    Get all folder recursively from a folder using Path and rglob.

    Args:
    - folder_path (str): The path to the folder.

    Returns:
    - list: A list of folder paths.
    """
    return [str(folder) for folder in Path(folder_path).rglob("*") if folder.is_dir()]


def read_class_mapping(class_mapping_csv):
    """
    Reads a CSV file mapping old class names to new class names.

    Args:
        class_mapping_csv (str):
            The path to the CSV file.

    Returns:
        dict: A dictionary mapping old class names to new class names.
    """
    class_mapping = {}
    with open(class_mapping_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 1:
                old_class = row[0]
                new_class = old_class
            elif len(row) >= 2:
                old_class, new_class = row[:2]
            class_mapping[old_class] = new_class
    return class_mapping


def get_corr_meta_names(img_path):
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    meta_filename = f".{base_filename}_meta.json"
    meta_path = os.path.join(os.path.dirname(img_path), meta_filename)
    return meta_path, meta_filename


def get_default_path(img_path):
    return img_path


def get_default_current_path(img_path):
    return img_path


def get_default_filename(img_path):
    return os.path.basename(img_path)


def get_default_group_id(img_path):
    return os.path.dirname(img_path).replace(os.path.sep, "_")


def get_default_image_size(img_path):
    with Image.open(img_path) as img:
        return img.size


def get_default_metadata(img_path, warn=False):
    img_path = os.path.abspath(img_path)
    # If metadata doesn't exist,
    # warn the user and generate default metadata
    if warn:
        print(
            f"File {img_path} does not have corresponding metadata. "
            "Generate default metadata for it."
        )
    meta_data = {
        "path": get_default_path(img_path),
        "current_path": get_default_current_path(img_path),
        "filename": get_default_filename(img_path),
        "group_id": get_default_group_id(img_path),
        "image_size": get_default_image_size(img_path),
    }
    return meta_data


def get_or_generate_metadata(img_path, warn=False, overwrite_path=False):
    img_path = os.path.abspath(img_path)
    meta_path, _ = get_corr_meta_names(img_path)
    updated = False

    # If metadata exists, load it
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)

        # Check for missing fields and update them
        if "path" not in meta_data or (
            overwrite_path and meta_data["path"] != img_path
        ):
            meta_data["path"] = img_path
            updated = True
        if "current_path" not in meta_data or meta_data["current_path"] != img_path:
            meta_data["current_path"] = img_path
            updated = True
        if "filename" not in meta_data:
            meta_data["filename"] = get_default_filename(img_path)
            updated = True
        if "group_id" not in meta_data:
            meta_data["group_id"] = get_default_group_id(img_path)
            updated = True
        if "image_size" not in meta_data:
            meta_data["image_size"] = get_default_image_size(img_path)
            updated = True
    else:
        meta_data = get_default_metadata(img_path, warn)
        updated = True
    if updated:
        with open(meta_path, "w") as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    return meta_data


def get_corr_ccip_names(img_path):
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    ccip_filename = f".{base_filename}_ccip.npy"
    ccip_path = os.path.join(os.path.dirname(img_path), ccip_filename)
    return ccip_path, ccip_filename


# TODO: Replace the use of this with construct_aux_files_dict
def get_related_paths(img_path):
    meta_path, _ = get_corr_meta_names(img_path)
    ccip_path, _ = get_corr_ccip_names(img_path)
    res = [meta_path, ccip_path]
    base_filename = os.path.splitext(img_path)[0]
    for ext in [".tags", ".processed_tags", ".characters"]:
        related_path = f"{base_filename}{ext}"
        res.append(related_path)
    return res
