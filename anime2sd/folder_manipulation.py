import os
import cv2
import json
import random
import logging
from pathlib import Path
from tqdm import tqdm


def get_files_recursively(folder_path):
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


def save_characters_to_meta(crop_dir):
    """
    Save character information to metadata files in the crop directory.

    Parameters:
    - crop_dir: Directory containing classified character folders.
    """

    encountered_paths = set()  # To keep track of paths encountered in this run

    # Iterate over each folder in the crop directory
    for folder_name in os.listdir(crop_dir):
        char_name = '_'.join(folder_name.split('_')[1:])
        if char_name == 'noise':
            continue
        folder_path = os.path.join(crop_dir, folder_name)

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

            # Construct the path to the corresponding metadata file
            meta_file_path = os.path.join(
                folder_path, f".{img_name}_meta.json")

            # If the metadata file exists, load it
            # otherwise initialize an empty dictionary
            if os.path.exists(meta_file_path):
                with open(meta_file_path, 'r') as meta_file:
                    meta_data = json.load(meta_file)
            else:
                meta_data = {}

            # Update the characters field
            meta_data['characters'] = [char_name]

            # Check for the 'path' field and update it
            if 'path' in meta_data:
                original_path = meta_data['path']
                orig_path_base = os.path.basename(original_path).split('.')[0]
                original_meta_path = os.path.join(
                    os.path.dirname(original_path),
                    f".{orig_path_base}_meta.json")

                # If the original metadata file exists,
                # update its characters field
                if os.path.exists(original_meta_path):
                    with open(original_meta_path, 'r') as orig_meta_file:
                        orig_meta_data = json.load(orig_meta_file)
                else:
                    orig_meta_data = {}

                # Initialize characters list
                # if the path hasn't been encountered in this run
                if original_path not in encountered_paths:
                    orig_meta_data['characters'] = []
                    encountered_paths.add(original_path)

                # Append the character name if it's not already in the list
                if char_name not in orig_meta_data['characters']:
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


def save_image(img, path, ext):
    """
    Save the image based on the provided extension.
    Adjusts the path to match the extension if necessary.
    """
    # Adjust the path to match the provided extension
    base_path = os.path.splitext(path)[0]
    adjusted_path = base_path + ext

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


def resize_character_images(crop_dir, full_dir, dst_dir,
                            max_size, ext, n_nocharacter_frames):
    # Ensure destination directories exist
    os.makedirs(os.path.join(dst_dir, 'cropped'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'full'), exist_ok=True)

    logging.log(f'Processing images from {crop_dir} ...')
    # Process images in crop_dir
    for img_file in tqdm(os.listdir(crop_dir)):
        img_path = os.path.join(crop_dir, img_file)
        meta_path = os.path.join(
            crop_dir, f".{os.path.splitext(img_file)[0]}_meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                meta_data = json.load(meta_file)

            if 'characters' in meta_data and meta_data['characters']:
                original_path = meta_data['path']
                original_img = cv2.imread(original_path)
                cropped_img = cv2.imread(img_path)

                if cropped_img.size > 0.5 * original_img.size:
                    continue

                resized_img = resize_image(cropped_img, max_size)
                save_path = os.path.join(dst_dir, 'cropped', img_file)
                save_image(resized_img, save_path, ext)

    logging.log(f'Processing images from {full_dir} ...')
    # Process images in full_dir
    nocharacter_frames = []
    for img_file in tqdm(get_files_recursively(full_dir)):
        img_path = os.path.join(full_dir, img_file)
        base_name = os.path.basename(original_path).split('.')[0]
        meta_path = os.path.join(
            full_dir, f".{base_name}_meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                meta_data = json.load(meta_file)

            if 'characters' in meta_data and meta_data['characters']:
                full_img = cv2.imread(img_path)
                resized_img = resize_image(full_img, max_size)
                save_path = os.path.join(dst_dir, 'full',
                                         os.path.basename(img_file))
                save_image(resized_img, save_path, ext)
        else:
            nocharacter_frames.append(img_path)

    # Randomly select n_nocharacter_frames and save
    if n_nocharacter_frames < len(nocharacter_frames):
        selected_frames = random.sample(
            nocharacter_frames, n_nocharacter_frames)
    else:
        selected_frames = nocharacter_frames
    for frame in selected_frames:
        img = cv2.imread(frame)
        resized_img = resize_image(img, max_size)
        save_path = os.path.join(dst_dir, 'full', os.path.basename(frame))
        save_image(resized_img, save_path, ext)
