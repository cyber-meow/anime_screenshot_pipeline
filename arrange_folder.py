import argparse
import cv2
import os
import json
import shutil

from tqdm import tqdm
from pathlib import Path

import numpy as np


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def get_folder_name(folder_type, info_dict, args):

    valid_folder_types = [
        'n_people',
        'n_faces',
        'fh_ratio',
        'n_characters',
        'character',
    ]
    assert folder_type in valid_folder_types,\
        f'invalid folder type {folder_type}'
    if folder_type == 'n_people':
        count = info_dict['n_people']
        suffix = 'person' if count == 1 else 'people'
        return f'{count}_{suffix}'
    elif folder_type == 'n_faces':
        count = info_dict['n_faces']
        suffix = 'face' if count == 1 else 'faces'
        return f'{count}_{suffix}'
    elif folder_type == 'n_characters':
        characters = sorted(list(set(info_dict['characters'])))
        for to_remove in ['unknown', 'ood']:
            characters = list(filter(
                lambda item: item != to_remove, characters))
        n_character = len(characters)
        if n_character >= args.max_character_number:
            return f'{args.max_character_number}+characters'
        suffix = 'character' if n_character == 1 else 'characters'
        return f'{n_character}_{suffix}'
    elif folder_type == 'character':
        characters = sorted(list(set(info_dict['characters'])))
        for to_remove in ['unknown', 'ood']:
            characters = list(filter(
                lambda item: item != to_remove, characters))
        if len(characters) == 0:
            return 'character_others'
        return '+'.join(sorted(characters))
    elif folder_type == 'fh_ratio':
        fh_ratio = min(int(info_dict['fh_ratio'] * 100), 99)
        folder_range = args.face_ratio_folder_range
        lb = fh_ratio // folder_range * folder_range
        ub = lb + folder_range
        return f'face_height_ratio_{lb}-{ub}'


def get_dst_dir(path, args):

    if args.keep_src_structure:
        dst_dir = os.path.dirname(path).replace(
            args.src_dir, args.dst_dir)
    else:
        dst_dir = args.dst_dir
    if args.format == '':
        return dst_dir, None

    path_noext = os.path.splitext(path)[0]
    with open(f'{path_noext}.json', 'r') as f:
        info_dict = json.load(f)

    folder_types = args.format.split('/')
    character_folder = None
    for folder_type in folder_types:
        folder_name = get_folder_name(folder_type, info_dict, args)
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
        if extension.lower() in [".png", ".jpg", ".jpeg", ".gif"]:
            # If it is, increment the count
            count += 1
    return count


def move_aux_files(old_path, new_path, move_file):

    old_path_noext = os.path.splitext(old_path)[0]
    new_path_noext = os.path.splitext(new_path)[0]

    original_aux_files = [
        old_path + '.tags',
        old_path_noext + '.facedata.json',  # for legacy
        old_path_noext + '.json',
        old_path_noext + '.txt',
    ]
    new_aux_files = [
        new_path + '.tags',
        new_path_noext + '.facedata.json',
        new_path_noext + '.json',
        new_path_noext + '.txt',
    ]

    for original_file, new_file in zip(original_aux_files, new_aux_files):
        if os.path.exists(original_file):
            if move_file:
                shutil.move(original_file, new_file)
            else:
                shutil.copy(original_file, new_file)


def merge_folder(character_comb_dict, min_image_per_comb):
    for comb in tqdm(character_comb_dict):
        files = character_comb_dict[comb]
        n_images = count_n_images(files)
        if n_images < min_image_per_comb:
            print(f'{comb} has fewer than {min_image_per_comb} images; '
                  + 'renamed as character_others')
            for file in files:
                new_path = file.replace(comb, 'character_others')
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(file, new_path)
                move_aux_files(file, new_path, move_file=True)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


# Written by chatgpt
def resize_image(image, max_size):

    height, width = image.shape[:2]
    if max_size > max(height, width):
        return image

    # Calculate the scaling factor
    scaling_factor = max_size / max(height, width)

    # Resize the image
    return cv2.resize(
        image, None, fx=scaling_factor, fy=scaling_factor,
        interpolation=cv2.INTER_AREA)


def main(args):

    print('processing.')
    output_extension = '.png'

    paths = get_files_recursively(args.src_dir)
    character_combination_dict = dict()

    for path in tqdm(paths):

        path_noext = os.path.splitext(path)[0]

        json_file = f'{path_noext}.json'
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f'Warning: {json_file} not found, skip')
            continue
        if 'n_faces' not in metadata:
            print(f'Warning: `n_faces` not found in {json_file}')
        n_faces = metadata['n_faces']
        if (n_faces < args.min_face_number
                or n_faces > args.max_face_number):
            continue

        dst_dir, character_folder = get_dst_dir(path, args)
        os.makedirs(dst_dir, exist_ok=True)
        new_path_noext = os.path.join(
            dst_dir, os.path.basename(path_noext))

        if args.move_file:
            new_path = os.path.join(dst_dir, os.path.basename(path))
            if path != new_path:
                shutil.move(path, new_path)
        else:
            try:
                image = cv2.imdecode(
                    np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
            except cv2.error as e:
                print(f'Error reading the image {path}: {e}')
                continue
            if image is None:
                print(f'Error reading the image {path}: get None')
                continue
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                # print(f'image has alpha. ignore: {path}')
                image = image[:, :, :3].copy()
            if args.max_image_size is not None:
                image = resize_image(image, args.max_image_size)
            output_extension = '.png'
            new_path = new_path_noext + output_extension
            _, buf = cv2.imencode(output_extension, image)
            with open(new_path, 'wb') as f:
                buf.tofile(f)

        if character_folder is not None:
            if character_folder in character_combination_dict:
                character_combination_dict[character_folder].append(new_path)
            else:
                character_combination_dict[character_folder] = [new_path]
        move_aux_files(path, new_path, args.move_file)

    if args.min_image_per_combination > 1:
        merge_folder(
            character_combination_dict, args.min_image_per_combination)
        remove_empty_folders(args.dst_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir', type=str,
        help='Directory to load images')
    parser.add_argument(
        '--dst_dir', type=str, default=None,
        help='Directory to save images; use source directory if not providied')
    parser.add_argument(
        '--move_file',
        action='store_true',
        help='Move the orignal image instead of saving a new one')
    parser.add_argument(
        '--max_image_size', type=int, default=None,
        help='Maximum size of the resulting image')
    parser.add_argument(
        '--min_face_number',
        type=int,
        default=1,
        help='The minimum number of faces an image should contain')
    parser.add_argument(
        '--max_face_number',
        type=int,
        default=10,
        help='The maximum number of faces an image can contain')
    parser.add_argument(
        '--keep_src_structure',
        action='store_true',
        help='If true the directory structure of the source directory is kept'
    )
    parser.add_argument(
        '--format', type=str, default='n_characters/character/fh_ratio',
        help='Description of the output directory hierarchy'
    )
    parser.add_argument('--count_singular', type=str, default='person')
    parser.add_argument('--count_plural', type=str, default='people')
    parser.add_argument(
        '--max_character_number', type=int, default=6,
        help='If have more than X characters put X+')
    parser.add_argument(
        '--min_image_per_combination', type=int, default=1,
        help='Put others instead of character name if nnumber of images '
        + 'of the character combination is smaller then this number')
    parser.add_argument(
        '--face_ratio_folder_range',
        type=int,
        default=25,
        help='The height ratio range of each separate folder')
    args = parser.parse_args()

    if args.dst_dir is None:
        args.dst_dir = args.src_dir

    main(args)
