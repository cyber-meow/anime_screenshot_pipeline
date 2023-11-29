import os
import json
import shutil
import logging
import random
import string
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List

from anime2sd.waifuc_customize import LocalSource, SaveExporter


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


def get_images_recursively(folder_path):
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


def get_related_paths(img_path):
    meta_path, _ = get_corr_meta_names(img_path)
    ccip_path, _ = get_corr_ccip_names(img_path)
    return [meta_path, ccip_path]


def construct_file_list(src_dir: str):
    """
    Construct a list of all files in the directory and checks for duplicates.

    Args:
        src_dir (str): The directory to search.
    Reurns:
        A list of all file paths in the directory.
    """
    all_files = {}
    for root, _, filenames in os.walk(src_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            if filename in all_files and filename != "multiply.txt":
                raise ValueError(f"Duplicate filename found: {filename}")
            all_files[filename] = path
    return all_files


def rearrange_related_files(src_dir: str):
    """
    Rearrange related files in some directory.

    Args:
        src_dir (src): The directory containing images and other files to rearrange.
    """
    all_files = construct_file_list(src_dir)
    image_files = get_images_recursively(src_dir)

    logging.info("Arranging related files ...")
    for img_path in tqdm(image_files, desc="Rearranging related files"):
        related_paths = get_related_paths(img_path)
        for related_path in related_paths:
            # If the related file does not exist in the expected location
            if not os.path.exists(related_path):
                # Search for the file in the all_files dictionary
                found_path = all_files.get(os.path.basename(related_path))
                if found_path is None:
                    if related_path.endswith("json"):
                        logging.warning(f"No related file found for {related_path}")
                        meta_data = get_default_metadata(img_path)
                        with open(related_path, "w") as f:
                            json.dump(meta_data, f)
                else:
                    # Move the found file to the expected location
                    shutil.move(found_path, related_path)
                    logging.info(
                        f"Moved related file from {found_path} " f"to {related_path}"
                    )


def load_metadata_from_aux(
    src_dir: str, load_aux: List[str], save_aux: List[str], overwrite_path: bool
) -> None:
    """
    Load metadata from auxiliary data and export it with potential modifications.

    This function loads metadata from a source directory, potentially modifies it,
    and then saves it back to the same directory.
    It utilizes auxiliary data specified in 'load_aux' and 'save_aux' lists.

    Args:
        src_dir (str): The source directory from which to load the metadata.
        load_aux (List[str]): A list of auxiliary data attributes to be loaded.
        save_aux (List[str]): A list of auxiliary data attributes to be saved.
        overwrite_path (bool):
            Flag to indicate if the path in the metadata should be overwritten.
    """
    logging.info("Load metadata from auxiliary data ...")
    source = LocalSource(src_dir, load_aux=load_aux, overwrite_path=overwrite_path)
    source.export(
        SaveExporter(
            src_dir,
            no_meta=False,
            save_caption=True,
            save_aux=save_aux,
            in_place=True,
        )
    )
