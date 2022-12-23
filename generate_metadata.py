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


def json_to_description(file_path, args):

    info_dict = {
        'general': args.general_description,
    }

    if args.retrieve_description_from_filename:
        to_prepend = os.path.basename(file_path).split('_')[:2]
        info_dict['general'] += ' ' + ''.join(to_prepend)
    if args.no_face:
        return info_dict

    filename_noext = os.path.splitext(file_path)[0]
    json_file = filename_noext + '.facedata.json'
    with open(json_file, 'r') as f:
        facedata = json.load(f)
        # characters = [char.replace('Korosaki', 'Kurosaki')
        #               for char in facedata['characters']]
        # facedata['characters'] = characters

    # filepath is of the form n_faces/face_height_ratio/character/X.png
    if args.use_character_folder:
        parentdir, characters = os.path.split(os.path.dirname(file_path))
        characters = list(set(characters.split('+')))
    else:
        parentdir = os.path.dirname(file_path)
        characters = facedata['characters']
    for to_remove in ['unknown', 'ood']:
        if to_remove in characters:
            characters.remove(to_remove)

    if args.use_count_folder:
        count = os.path.basename(os.path.dirname(parentdir))
        count = count.split('_')[0]
    else:
        count = str(facedata[args.count_description])

    mark_face_position = args.always_mark_facepos
    if count.isnumeric() and int(count) == 1:
        mark_face_position = True
    else:
        # If detection and classification is good
        if (str(count) == str(facedata[args.count_description])
                and sorted(characters) == sorted(facedata['characters'])):
            mark_face_position = True
            # Use the order in face data that corresponds to face position
            characters = facedata['characters']

    info_dict['count'] = count
    info_dict['characters'] = characters

    if mark_face_position:
        face_position_descr = []
        for rel_pos in facedata['rel_pos']:
            left, top, right, bottom = rel_pos
            left = int(left * 100)
            right = int(right * 100)
            top = int(top * 100)
            bottom = int(bottom * 100)
            face_v_position_info = f'fvp {top} {bottom}'
            face_h_position_info = f' fhp {left} {right}'
            face_position_descr.append(
                face_v_position_info + face_h_position_info)
        info_dict['facepos'] = face_position_descr
    return info_dict


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
                del(tags[:k])
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
        "--use_character_folder", action="store_true",
        help="Use character folder structure for characters")
    parser.add_argument(
        "--use_count_folder", action="store_true",
        help="Use count folder structure for number of people etc")
    parser.add_argument(
        "--always_mark_facepos", action="store_true",
        help="Always save face position data if provided")
    parser.add_argument(
        "--count_description", default='n_faces',
        help="The dictionary key to retrieve count information")
    parser.add_argument(
        "--general_description", default='anishot',
        help="General description of the files")
    parser.add_argument(
        "--retrieve_description_from_filename", action='store_true')
    parser.add_argument(
        "--use_tags", action='store_true',
        help="Use tag files to retrieve information")
    parser.add_argument(
        "--use_tags_for_count", action='store_true',
        help="Use tag files to count people and add as count")
    parser.add_argument(
        "--remove_before_girl", action='store_true',
        help="Remove the tags that appear before [k]girl(s)")
    parser.add_argument(
        "--no_face", action='store_true',
        help="Ignore face data for data with no face")
    args = parser.parse_args()

    file_paths = get_files_recursively(args.src_dir)
    for file_path in tqdm(file_paths):
        basic_info = json_to_description(file_path, args)
        if args.use_tags or args.use_tags_for_count:
            tags_file = file_path + '.tags'
            if os.path.exists(tags_file):
                with open(tags_file, 'r') as f:
                    lines = f.readlines()
                if args.use_tags:
                    basic_info = retrieve_tag_info(
                        basic_info, lines, args.remove_before_girl)
                if args.use_tags_for_count:
                    n_people = get_npeople_from_tags(lines)
                    basic_info['count'] = n_people
            else:
                print(f'Warning: tags file {tags_file} not found')
        filename_noext = os.path.splitext(file_path)[0]
        savefile = filename_noext + '.json'
        with open(savefile, 'w') as f:
            json.dump(basic_info, f)
