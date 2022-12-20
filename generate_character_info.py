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


def json_to_description(file_path, use_character_folder=True):
    filename_noext = os.path.splitext(file_path)[0]
    json_file = filename_noext + '_facedata.json'
    with open(json_file, 'r') as f:
        facedata = json.load(f)
        # characters = [char.replace('Korosaki', 'Kurosaki')
        #               for char in facedata['characters']]
        # facedata['characters'] = characters
    # with open(json_file, 'w') as f:
    #     json.dump(facedata, f)
    # filepath is of the form n_faces/face_height_ratio/character/X.png

    if use_character_folder:
        parentdir, characters = os.path.split(os.path.dirname(file_path))
        _, n_faces = os.path.split(os.path.dirname(parentdir))
        n_faces = n_faces.split('_')[0]
        characters = characters.split('+')
    else:
        n_faces = facedata['n_faces']
        characters = facedata['characters']

    mark_face_position = False
    if n_faces == 1:
        caption = characters[0]
        caption = 'solo, ' + caption
        mark_face_position = True
    else:
        # If detection and classification is good
        if (n_faces == facedata['n_faces']
                and characters == sorted(facedata['characters'])):
            mark_face_position = True
        caption = ', '.join(characters)
        caption = f'{n_faces} people, ' + caption

    if mark_face_position:
        caption = caption + '\n'
        for rel_pos in facedata['rel_pos']:
            left, top, right, bottom = rel_pos
            left = int(left * 100)
            right = int(right * 100)
            top = int(top * 100)
            bottom = int(bottom * 100)
            face_v_position_info = f'fvp {top} {bottom}'
            face_h_position_info = f' fhp {left} {right}\n'
            caption += face_v_position_info + face_h_position_info
    return caption


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory to load images")
    parser.add_argument(
        "--use_character_folder", action="store_true",
        help="use character folder structure to determine number "
        + "of people and characters")
    args = parser.parse_args()

    file_paths = get_files_recursively(args.src_dir)
    for file_path in tqdm(file_paths):
        basic_description = json_to_description(
            file_path, args.use_character_folder)
        filename_noext = os.path.splitext(file_path)[0]
        savefile = filename_noext + '.characterinfo'
        with open(savefile, 'w') as f:
            f.write(basic_description)
