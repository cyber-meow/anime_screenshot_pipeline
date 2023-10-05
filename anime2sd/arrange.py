import os
import json
import shutil
import logging
from tqdm import tqdm

from anime2sd.basics import get_images_recursively, get_corr_meta_names


def is_aux_file(img_file, candidate_file):
    """
    Check if candidate_file is an auxiliary file for img_file.

    :param img_file: The image file name.
    :param candidate_file: The candidate file name.
    :return: True if candidate_file is an auxiliary file for
        img_file, False otherwise.
    """
    img_base, _ = os.path.splitext(img_file)

    candidate_base, _ = os.path.splitext(candidate_file)

    # Check the conditions
    # not dedaling with ccip cache
    if (candidate_base == img_base or
            candidate_base.startswith(f".{img_base}_meta")):
        return True

    return False


def move_img_with_aux(img_path, dst_dir):
    """
    Move image along with its auxiliary files to a destination directory.

    :param img_path: Path to the image file.
    :param dst_dir: Destination directory.
    :return: None
    """
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Get the directory containing the image file
    src_dir = os.path.dirname(img_path)

    # List all files in the source directory
    for candidate_file in os.listdir(src_dir):
        # Construct the full path of candidate_file
        candidate_path = os.path.join(src_dir, candidate_file)

        if (candidate_path == img_path
                or is_aux_file(os.path.basename(img_path), candidate_file)):
            if os.path.dirname(candidate_path) != dst_dir:
                shutil.move(candidate_path, dst_dir)


def get_folder_name(folder_type, info_dict, max_character_number):

    valid_folder_types = [
        'n_people',
        'n_characters',
        'character',
    ]
    assert folder_type in valid_folder_types,\
        f'invalid folder type {folder_type}'
    if folder_type == 'n_people':
        count = info_dict['n_people']
        suffix = 'person' if count == 1 else 'people'
        return f'{count}_{suffix}'
    elif folder_type == 'n_characters':
        characters = info_dict.get('characters', [])
        if len(characters) > 0 and type(characters[0]) is list:
            characters = [character[0] for character in characters]
        characters = sorted(list(set(characters)))
        n_character = len(characters)
        if n_character >= max_character_number:
            return f'{max_character_number}+_characters'
        suffix = 'character' if n_character == 1 else 'characters'
        return f'{n_character}_{suffix}'
    elif folder_type == 'character':
        characters = info_dict.get('characters', [])
        if len(characters) > 0 and type(characters[0]) is list:
            characters = [character[0] for character in characters]
        characters = sorted(list(set(characters)))
        if len(characters) == 0:
            return 'character_others'
        character_folder = '+'.join(sorted(characters))
        # Cannot have folder of too long name
        if len(character_folder) >= 100:
            return 'character_others'
        else:
            return character_folder.replace('.', '')


def get_dst_dir(info_dict, dst_dir, arrange_format, max_character_number):

    if arrange_format == '':
        return dst_dir, None

    folder_types = arrange_format.split('/')
    character_folder = None
    for folder_type in folder_types:
        folder_name = get_folder_name(
            folder_type, info_dict, max_character_number)
        dst_dir = os.path.join(dst_dir, folder_name)
        if folder_type == 'character':
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


def merge_folder(character_comb_dict, min_images_per_comb):
    for comb in tqdm(character_comb_dict):
        files = character_comb_dict[comb]
        n_images = count_n_images(files)
        if n_images < min_images_per_comb:
            logging.info(
                f'{comb} has fewer than {min_images_per_comb} images; '
                + 'renamed as character_others')
            for file in files:
                new_path = file.replace(comb, 'character_others')
                dst_dir = os.path.dirname(new_path)
                move_img_with_aux(file, dst_dir)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def arrange_folder(src_dir,
                   dst_dir,
                   arrange_format,
                   max_character_number,
                   min_images_per_combination):

    img_paths = get_images_recursively(src_dir)
    character_combination_dict = dict()

    for img_path in tqdm(img_paths):

        meta_file_path, _ = get_corr_meta_names(img_path)

        if os.path.exists(meta_file_path):
            with open(meta_file_path, 'r') as meta_file:
                meta_data = json.load(meta_file)
        else:
            raise ValueError(f'Metadata unfound for {img_path}')

        img_dst_dir, character_folder = get_dst_dir(
            meta_data, dst_dir, arrange_format, max_character_number)

        move_img_with_aux(img_path, img_dst_dir)

        new_path = os.path.join(img_dst_dir, os.path.basename(img_path))
        if character_folder is not None:
            if character_folder in character_combination_dict:
                character_combination_dict[character_folder].append(new_path)
            else:
                character_combination_dict[character_folder] = [new_path]

    if min_images_per_combination > 1:
        merge_folder(
            character_combination_dict, min_images_per_combination)
    remove_empty_folders(src_dir)
    remove_empty_folders(dst_dir)
