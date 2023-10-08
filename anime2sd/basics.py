import os
import json
import shutil
import logging
from pathlib import Path
from tqdm import tqdm


def get_images_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]',
        '*.[Jj][Pp][Gg]',
        '*.[Jj][Pp][Ee][Gg]',
        '*.[Ww][Ee][Bb][Pp]',
        '*.[Gg][Ii][Ff]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def default_metadata(img_path):
    meta_data = {'path': img_path,
                 'current_path': img_path,
                 'filename': os.path.basename(img_path),
                 'group_id': os.path.dirname(
                     img_path
                 ).replace(os.path.sep, '_')}
    return meta_data


def get_corr_meta_names(img_path):
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    meta_filename = f".{base_filename}_meta.json"
    meta_path = os.path.join(os.path.dirname(img_path), meta_filename)
    return meta_path, meta_filename


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


def construct_file_list(src_dir):
    """
    Construct a list of all files in the directory and checks for duplicates.

    :param classified_dir: The directory to search.
    :return: A list of all file paths in the directory.
    """
    all_files = {}
    for root, _, filenames in os.walk(src_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            if filename in all_files and filename != 'multiply.txt':
                raise ValueError(f"Duplicate filename found: {filename}")
            all_files[filename] = path
    return all_files


def rearrange_related_files(src_dir):
    """
    Rearrange related files in some directory.

    :param src_dir: The directory containing images
        and other files to rearrange.
    """
    all_files = construct_file_list(src_dir)
    image_files = get_images_recursively(src_dir)

    logging.info('Arranging related files ...')
    for img_path in tqdm(image_files, desc="Rearranging related files"):
        related_paths = get_related_paths(img_path)
        for related_path in related_paths:
            # If the related file does not exist in the expected location
            if not os.path.exists(related_path):
                # Search for the file in the all_files dictionary
                found_path = all_files.get(os.path.basename(related_path))
                if found_path is None:
                    if related_path.endswith('json'):
                        logging.warning(
                            f"No related file found for {related_path}")
                        meta_data = default_metadata(img_path)
                        with open(related_path, 'w') as f:
                            json.dump(meta_data, f)
                else:
                    # Move the found file to the expected location
                    shutil.move(found_path, related_path)
                    logging.info(
                        f"Moved related file from {found_path} "
                        f"to {related_path}")
