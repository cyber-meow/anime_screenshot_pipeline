import os
import json
import argparse

from tqdm import tqdm
from pathlib import Path


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
        '*.[Gg][Ii][Ff]'
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def get_basic_metadata(file_path, args):

    filename_noext = os.path.splitext(file_path)[0]
    json_file = filename_noext + '.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = dict()

    metadata['general'] = args.general_description
    if args.retrieve_description_from_filename:
        to_prepend = os.path.basename(file_path).split('_')[:2]
        metadata['general'] += ' ' + ''.join(to_prepend)

    return metadata


def to_list(line):
    items = [item.strip() for item in line.split(',')]
    return items


def retrieve_tag_info(basic_info, tags_content, remove_before_girl):

    for line in tags_content:
        # If tags contain character information use it instead
        if line.startswith('character: '):
            characters = to_list(line.lstrip('character:'))
            for to_remove in ['unknown', 'ood']:
                if to_remove in characters:
                    characters.remove(to_remove)
            basic_info['characters'] = characters
        if line.startswith('copyright: '):
            basic_info['copyright'] = to_list(line.lstrip('copyright:'))
        if line.startswith('artist: '):
            basic_info['artist'] = to_list(line.lstrip('artist:').strip())
        elif line.startswith('general: '):
            tags = to_list(line.lstrip('general: '))
        # This is the case when there is a single line of tags
        else:
            tags = to_list(line.strip())

    if remove_before_girl:
        glist = ['1girl', '2girls', '3girls', '4girls', '5girls', '6+girls']
        for k, tag in enumerate(tags):
            if tag in glist:
                del (tags[:k])
                break

    basic_info['tags'] = tags
    return basic_info


def get_npeople_from_tags(tags_content):

    girl_dictinary = {
        '1girl': 1,
        '6+girls': 6,
    }
    boy_dictinary = {
        '1boy': 1,
        '6+boys': 6,
    }
    for k in range(2, 6):
        girl_dictinary[f'{k}girls'] = k
        boy_dictinary[f'{k}boys'] = k

    n_girls = 0
    n_boys = 0
    for line in tags_content:
        for key in girl_dictinary:
            if key in line:
                n_girls = max(n_girls, girl_dictinary[key])
        for key in boy_dictinary:
            if key in line:
                n_boys = max(n_boys, boy_dictinary[key])
    n_people = n_girls + n_boys
    if n_people >= 6:
        n_people = 'many'
    return n_people


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory to load images")
    parser.add_argument(
        "--general_description", default='anishot',
        help="General description of the files")
    parser.add_argument(
        "--retrieve_description_from_filename", action='store_true')
    parser.add_argument(
        "--use_tags", action='store_true',
        help="Use tag files to retrieve information")
    parser.add_argument(
        "--remove_before_girl", action='store_true',
        help="Remove the tags that appear before [k]girl(s)")
    args = parser.parse_args()

    file_paths = get_files_recursively(args.src_dir)
    for file_path in tqdm(file_paths):
        metadata = get_basic_metadata(file_path, args)
        if args.use_tags:
            tags_file = file_path + '.tags'
            if os.path.exists(tags_file):
                with open(tags_file, 'r') as f:
                    lines = f.readlines()
                metadata = retrieve_tag_info(
                    metadata, lines, args.remove_before_girl)
                n_people = get_npeople_from_tags(lines)
                metadata['n_people'] = n_people
            else:
                print(f'Warning: tags file {tags_file} not found')
        filename_noext = os.path.splitext(file_path)[0]
        savefile = filename_noext + '.json'
        with open(savefile, 'w') as f:
            json.dump(metadata, f)
