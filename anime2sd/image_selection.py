import os
import cv2
import json
import random
import logging
import shutil
from tqdm import tqdm

from anime2sd.basics import get_images_recursively
from anime2sd.basics import get_corr_meta_names, default_metadata
from anime2sd.basics import get_related_paths


def construct_file_list(classified_dir):
    """
    Construct a list of all files in the directory and checks for duplicates.

    :param classified_dir: The directory to search.
    :return: A list of all file paths in the directory.
    """
    all_files = {}
    for root, _, filenames in os.walk(classified_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            if filename in all_files:
                raise ValueError(f"Duplicate filename found: {filename}")
            all_files[filename] = path
    return all_files


def rearrange_related_files(classified_dir):
    """
    Rearrange related files in the classified directory.

    :param classified_dir: The directory containing classified images.
    """
    all_files = construct_file_list(classified_dir)
    image_files = get_images_recursively(classified_dir)

    logging.info('Arranging related files ...')
    for img_path in tqdm(image_files, desc="Rearranging related files"):
        related_paths = get_related_paths(img_path)
        for related_path in related_paths:
            # If the related file does not exist in the expected location
            if not os.path.exists(related_path):
                # Search for the file in the all_files dictionary
                found_path = all_files.get(os.path.basename(related_path))
                if found_path is None:
                    logging.warning(
                        f"No related file found for {related_path}")
                    if related_path.endswith('json'):
                        meta_data = default_metadata(img_path)
                        with open(related_path, 'w') as f:
                            json.dump(meta_data, f)
                else:
                    # Move the found file to the expected location
                    shutil.move(found_path, related_path)
                    logging.info(
                        f"Moved related file from {found_path} "
                        f"to {related_path}")


def parse_char_name(folder_name):
    if '_' in folder_name:
        parts = folder_name.split('_')
        if parts[0].strip('-').isdigit():
            return '_'.join(parts[1:])
    return folder_name


def save_characters_to_meta(classified_dir):
    """
    Save character information to metadata files in the crop directory.

    Parameters:
    - classified_dir: Directory containing classified character folders.
    """

    encountered_paths = set()  # To keep track of paths encountered in this run

    logging.info('Saving characters to metadata ...')
    # Iterate over each folder in the classified directory
    for folder_name in tqdm(os.listdir(classified_dir)):
        char_name = parse_char_name(folder_name)
        folder_path = os.path.join(classified_dir, folder_name)

        # Ensure it's a directory
        if not os.path.isdir(folder_path):
            continue

        # Iterate over each image file in the folder
        for img_file in os.listdir(folder_path):
            img_name, img_ext = os.path.splitext(img_file)

            # Ensure it's an image file
            if img_ext.lower() not in [
                    '.png', '.jpg', '.jpeg', '.webp', '.gif']:
                continue

            img_path = os.path.join(folder_path, img_file)
            # Construct the path to the corresponding metadata file
            meta_file_path, _ = get_corr_meta_names(img_path)

            # If the metadata file exists, load it
            # otherwise initialize an empty dictionary
            if os.path.exists(meta_file_path):
                with open(meta_file_path, 'r') as meta_file:
                    meta_data = json.load(meta_file)
            else:
                meta_data = default_metadata(img_path)
                logging.warning(
                    f'Cropped file {img_path} does not have '
                    + 'corresponding metadata')

            # Update the characters field
            if char_name.startswith('noise') or char_name.startswith('Noise'):
                # This ensures that we overwrite old information
                meta_data['characters'] = []
            else:
                meta_data['characters'] = [char_name]

            # Check for the 'path' field and update it
            if 'path' in meta_data:
                original_path = meta_data['path']
                original_meta_path, _ = get_corr_meta_names(original_path)

                # If the original metadata file exists,
                # update its characters field
                if os.path.exists(original_meta_path):
                    with open(original_meta_path, 'r') as orig_meta_file:
                        orig_meta_data = json.load(orig_meta_file)
                else:
                    orig_meta_data = default_metadata(original_path)

                # Initialize characters list
                # if the path hasn't been encountered in this run
                if original_path not in encountered_paths:
                    orig_meta_data['characters'] = []
                    encountered_paths.add(original_path)

                # Append the character name if it's not already in the list
                # and is not noise
                if (char_name not in orig_meta_data['characters']
                        and not char_name.lower().startswith('noise')):
                    orig_meta_data['characters'].append(char_name)

                # Save the updated original metadata
                with open(original_meta_path, 'w') as orig_meta_file:
                    json.dump(orig_meta_data, orig_meta_file, indent=4)

            # Save the updated metadata for the cropped image
            with open(meta_file_path, 'w') as meta_file:
                json.dump(meta_data, meta_file, indent=4)


def resize_image(image, max_size):

    height, width = image.shape[:2]
    if max_size > min(height, width):
        return image

    # Calculate the scaling factor
    scaling_factor = max_size / min(height, width)

    # Resize the image
    return cv2.resize(
        image, None, fx=scaling_factor, fy=scaling_factor,
        interpolation=cv2.INTER_AREA)


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

    # Save the image
    if ext == '.webp':
        try:
            _, buf = cv2.imencode(ext, img, [cv2.IMWRITE_WEBP_QUALITY, 95])
        except cv2.error as e:
            print(f'Error encoding the image {adjusted_path}: {e}')
            return
        # Save the encoded image
        with open(adjusted_path, 'wb') as f:
            buf.tofile(f)
    else:
        # For other formats, we can use imwrite directly
        cv2.imwrite(adjusted_path, img)

    # Copy the corresponding metadata file
    meta_path, meta_filename = get_corr_meta_names(img_path)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as meta_file:
            meta_data = json.load(meta_file)
        meta_data['filename'] = meta_data['filename'].replace('.png', ext)
        meta_data['type'] = image_type

        # Save the updated metadata with new extension
        with open(os.path.join(save_dir, meta_filename), 'w') as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    else:
        raise ValueError('All metadata must exist before resizing to dst')


def copy_image_and_meta(img_path, save_dir, image_type):
    shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))
    # Copy the corresponding metadata file
    meta_path, meta_filename = get_corr_meta_names(img_path)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as meta_file:
            meta_data = json.load(meta_file)
        meta_data['type'] = image_type

        # Save the updated metadata with new extension
        with open(os.path.join(save_dir, meta_filename), 'w') as meta_file:
            json.dump(meta_data, meta_file, indent=4)
    else:
        raise ValueError('All metadata must exist before resizing to dst')


def resize_character_images(classified_dir, full_dir, dst_dir,
                            max_size, ext, image_type,
                            n_nocharacter_frames, to_resize=True):

    # Process images in classified_dir
    logging.info(f'Processing images from {classified_dir} ...')
    save_dir = os.path.join(dst_dir, 'cropped')
    os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm(get_images_recursively(classified_dir)):
        meta_path, _ = get_corr_meta_names(img_path)

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                meta_data = json.load(meta_file)

            if 'characters' in meta_data and meta_data['characters']:
                original_path = meta_data['path']
                original_img = cv2.imread(original_path)
                cropped_img = cv2.imread(img_path)

                if cropped_img.size > 0.5 * original_img.size:
                    continue

                if to_resize:
                    resized_img = resize_image(cropped_img, max_size)
                    save_image_and_meta(
                        resized_img, img_path, save_dir, ext, image_type)
                else:
                    copy_image_and_meta(img_path, save_dir, image_type)
        else:
            raise ValueError(
                'All the cropped files should have corresponding metadata')

    # Process images in full_dir
    logging.info(f'Processing images from {full_dir} ...')
    save_dir = os.path.join(dst_dir, 'full')
    os.makedirs(save_dir, exist_ok=True)
    nocharacter_frames = []
    for img_path in tqdm(get_images_recursively(full_dir)):
        meta_path, _ = get_corr_meta_names(img_path)

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                meta_data = json.load(meta_file)

            if 'characters' in meta_data and meta_data['characters']:

                if to_resize:
                    full_img = cv2.imread(img_path)
                    resized_img = resize_image(full_img, max_size)
                    save_image_and_meta(
                        resized_img, img_path, save_dir, ext, image_type)
                else:
                    copy_image_and_meta(img_path, save_dir, image_type)
            else:
                nocharacter_frames.append(img_path)
        else:
            meta_data = default_metadata(img_path)
            with open(meta_path, 'w') as meta_file:
                json.dump(meta_data, meta_file, indent=4)
            nocharacter_frames.append(img_path)

    # Process no character images
    # Randomly select n_nocharacter_frames and save
    save_dir = os.path.join(dst_dir, 'no_characters')
    os.makedirs(save_dir, exist_ok=True)

    if n_nocharacter_frames < len(nocharacter_frames):
        selected_frames = random.sample(
            nocharacter_frames, n_nocharacter_frames)
    else:
        selected_frames = nocharacter_frames

    logging.info(f'Copying {len(selected_frames)} no character images ...')

    for img_path in tqdm(selected_frames):
        if to_resize:
            img = cv2.imread(img_path)
            resized_img = resize_image(img, max_size)
            save_image_and_meta(resized_img, img_path,
                                save_dir, ext, image_type)
        else:
            copy_image_and_meta(img_path, save_dir, image_type)
