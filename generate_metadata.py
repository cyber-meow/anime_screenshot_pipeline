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
    filename_noext = os.path.splitext(file_path)[0]
    json_file = filename_noext + '.facedata.json'
    with open(json_file, 'r') as f:
        facedata = json.load(f)
        # characters = [char.replace('Korosaki', 'Kurosaki')
        #               for char in facedata['characters']]
        # facedata['characters'] = characters
    # with open(json_file, 'w') as f:
    #     json.dump(facedata, f)
    # filepath is of the form n_faces/face_height_ratio/character/X.png

    if args.use_character_folder:
        parentdir, characters = os.path.split(os.path.dirname(file_path))
        characters = characters.split('+')
    else:
        parentdir = os.path.dirname(file_path)
        characters = facedata['characters']

    if args.use_count_folder:
        count = os.path.basename(os.path.dirname(parentdir))
        count = count.split('_')[0]
    else:
        count = str(facedata[args.count_description])

    mark_face_position = False
    if count.isnumeric() and int(count) == 1:
        mark_face_position = True
    else:
        # If detection and classification is good
        if (str(count) == str(facedata[args.count_description])
                and sorted(characters) == sorted(facedata['characters'])):
            mark_face_position = True

    info_dict = {
        'count': count,
        'characters': characters,
        'general': args.general_description,
    }

    if args.retrieve_description_from_filename:
        to_prepend = os.path.basename(file_path).split('_')[:2]
        info_dict['general'] += ' ' + ''.join(to_prepend)

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


def retrieve_tag_info(basic_info, tags_content, formatted_tags):
    if formatted_tags:
        for line in tags_content:
            if line.startswith('copyright: '):
                basic_info['copyright'] = line.lstrip('copyright:').strip()
            if line.startswith('artist: '):
                basic_info['artist'] = line.lstrip('artist:').strip()
            if line.startswith('general: '):
                tags_descr = line.lstrip('general: ')
    else:
        # Tags in one line separated by ,
        tags_descr = tags_content[0]
    tags = tags_descr.split(',')
    basic_info['tags'] = [tag.strip() for tag in tags]
    return basic_info


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
        "--count_description", default='n_people',
        help="The dictionary key to retrieve count information")
    parser.add_argument(
        "--general_description", default='anishot',
        help="General description of the files")
    parser.add_argument(
        "--retrieve_description_from_filename", action='store_true')
    parser.add_argument(
        "--use_formatted_tags", action='store_true',
        help="Use formatted tag files for example downloaded from grabber")
    parser.add_argument(
        "--use_unformatted_tags", action='store_true',
        help="Use unformatted tag files for example generateed by tagger")
    args = parser.parse_args()
    assert not (args.use_formatted_tags and args.use_unformatted_tags)

    file_paths = get_files_recursively(args.src_dir)
    for file_path in tqdm(file_paths):
        basic_info = json_to_description(file_path, args)
        if args.use_formatted_tags or args.use_unformatted_tags:
            tags_file = file_path + '.tags'
            if os.path.exists(tags_file):
                with open(tags_file, 'r') as f:
                    lines = f.readlines()
                    basic_info = retrieve_tag_info(
                        basic_info, lines, args.use_formatted_tags)
            else:
                print(f'Warning: tags file {tags_file} not found')
        filename_noext = os.path.splitext(file_path)[0]
        savefile = filename_noext + '.json'
        with open(savefile, 'w') as f:
            json.dump(basic_info, f)
