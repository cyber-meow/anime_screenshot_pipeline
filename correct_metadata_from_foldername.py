import argparse
import os
import json

from tqdm import tqdm
from pathlib import Path


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def correct_metadata(path, path_format, character_list):
    json_file = os.path.splitext(path)[0] + '.json'
    if not os.path.exists(json_file):
        print(f'Warning: {json_file} unfound, skip')
        return
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    to_correct = path_format.split('/')
    dirname = os.path.dirname(path)
    for folder_type in reversed(to_correct):
        dirname, basename = os.path.split(dirname)
        correct_metadata_single(metadata, folder_type,
                                basename, character_list)
    with open(json_file, 'w') as f:
        json.dump(metadata, f)


def correct_metadata_single(
        metadata, folder_type, basename, character_list=None):
    if folder_type == '*':
        return
    elif folder_type == 'n_faces':
        if basename == '1face':
            metadata['n_faces'] = 1
        else:
            count = basename.rstrip('faces')
            if count.isnumeric():
                count = int(count)
            metadata['n_faces'] = count
    elif folder_type == 'n_people':
        if basename == '1person':
            metadata['n_people'] = 1
        else:
            count = basename.rstrip('people')
            if count.isnumeric():
                count = int(count)
            metadata['n_people'] = count
    elif folder_type == 'character':
        if basename in ['character_others', 'others']:
            return
        if basename == 'ood':
            characters = []
        else:
            characters = sorted(list(set(basename.split('+'))))
        for to_remove in ['unknown', 'ood']:
            if to_remove in characters:
                characters.remove(to_remove)
        if 'characters' in metadata:
            characters_in_meta = sorted(
                list(set(metadata['characters'])))
            for to_remove in ['unknown', 'ood']:
                if to_remove in characters_in_meta:
                    characters_in_meta.remove(to_remove)
            # No need for correction if metadata agrees with folder name
            # Use the original metadata preserve order that correspond to
            # that of facepos
            if characters == characters_in_meta:
                return
        if character_list is not None:
            for character in characters:
                assert character in character_list, \
                    f'Invalid character {character} for {basename}'
        metadata['characters'] = characters
    else:
        print(f'Warning: invalid folder type {folder_type}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir', type=str,
        help='Directory to load images')
    parser.add_argument(
        '--format', type=str, default='*/character',
        help='Description of the output directory hierarchy'
    )
    parser.add_argument(
        "--character_list", type=str, default=None,
        help="Txt file containing character names separated "
        + "by comma or new line")
    args = parser.parse_args()

    if args.character_list is not None:
        with open(args.character_list, 'r') as f:
            lines = f.readlines()
        character_list = []
        for line in lines:
            character_list.extend(line.strip().split(','))
        print(character_list)
    else:
        character_list = None

    paths = get_files_recursively(args.src_dir)

    for path in tqdm(paths):
        correct_metadata(path, args.format, character_list)
