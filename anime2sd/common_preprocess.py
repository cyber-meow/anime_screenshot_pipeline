import os
import json
import shutil
import logging
from tqdm import tqdm
from typing import List, Dict, Optional

from anime2sd.basics import (
    get_images_recursively,
    get_related_paths,
    get_default_metadata,
)
from anime2sd.waifuc_customize import LocalSource, SaveExporter, TagRenameAction


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


def rearrange_related_files(src_dir: str, logger: Optional[logging.Logger] = None):
    """
    Rearrange related files in some directory.

    Args:
        src_dir (src): The directory containing images and other files to rearrange.
        logger (Logger): A logger to use for logging.
    """
    if logger is None:
        logger = logging.getLogger()
    all_files = construct_file_list(src_dir)
    image_files = get_images_recursively(src_dir)

    logger.info("Arranging related files ...")
    for img_path in tqdm(image_files, desc="Rearranging related files"):
        related_paths = get_related_paths(img_path)
        for related_path in related_paths:
            # If the related file does not exist in the expected location
            if not os.path.exists(related_path):
                # Search for the file in the all_files dictionary
                found_path = all_files.get(os.path.basename(related_path))
                if found_path is None:
                    if related_path.endswith("json"):
                        logger.warning(f"No related file found for {related_path}")
                        meta_data = get_default_metadata(img_path)
                        with open(related_path, "w") as f:
                            json.dump(meta_data, f)
                else:
                    # Move the found file to the expected location
                    shutil.move(found_path, related_path)
                    logger.info(
                        f"Moved related file from {found_path} " f"to {related_path}"
                    )


def load_metadata_from_aux(
    src_dir: str,
    load_grabber_ext: Optional[str],
    load_aux: List[str],
    overwrite_path: bool,
    character_mapping: Optional[Dict[str, str]],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Load metadata from auxiliary data and export it with potential modifications.

    This function loads metadata from a source directory, potentially modifies it,
    and then saves it back to the same directory.

    Args:
        src_dir (str):
            The source directory from which to load the metadata.
        load_grabber_ext (Optional[str]):
            The extension of the grabber information files to be loaded.
        load_aux (List[str]):
            A list of auxiliary data attributes to be loaded.
        overwrite_path (bool):
            Flag to indicate if the path in the metadata should be overwritten.
        character_mapping (Optional[Dict[str, str]]):
            A mapping from old character names to new character names.
        logger (Logger): Logger to use for logging.
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info("Load metadata from auxiliary data ...")
    source = LocalSource(
        src_dir,
        load_grabber_ext=load_grabber_ext,
        load_aux=load_aux,
        overwrite_path=overwrite_path,
    )
    if character_mapping:
        # Renaming characters
        source = source.attach(
            TagRenameAction(character_mapping, fields=["characters"])
        )
    source.export(
        SaveExporter(
            src_dir,
            no_meta=False,
            in_place=True,
        )
    )
