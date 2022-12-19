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


def json_to_description(file_path):
    basename = os.path.splitext(file_path)[0]
    json_file = basename + '_facedata.json'
    with open(json_file, 'r') as f:
        facedata = json.load(f)
        facedata['characters'] = [
            charac.replace('Yiki', 'Yuki') for charac in facedata['characters']
        ]
    with open(json_file, 'w') as f:
        json.dump(facedata, f)
    # filepath is of the form n_faces/face_height_ratio/character/X.png
    basedir, characters = os.path.split(os.path.dirname(file_path))
    _, n_faces = os.path.split(os.path.dirname(basedir))
    n_faces = int(n_faces.split('_')[0])
    characters = characters.split('+')

    mark_face_position = False
    if n_faces == 1:
        caption = characters[0]
        caption = caption + ', solo'
        mark_face_position = True
    else:
        # If detection and classification is good
        if (n_faces == facedata['n_faces']
                and characters == sorted(facedata['characters'])):
            characters = facedata['characters']
            mark_face_position = True
        caption = ' '.join(characters)
        if n_faces < 10:
            caption = caption + f', {n_faces} people'
        else:
            caption = caption + f', many people'

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
    args = parser.parse_args()

    file_paths = get_files_recursively(args.src_dir)
    for file_path in tqdm(file_paths):
        basic_description = json_to_description(file_path)
        basename = os.path.splitext(file_path)[0]
        savefile = basename + '.characterdata'
        with open(savefile, 'w') as f:
            f.write(basic_description)
