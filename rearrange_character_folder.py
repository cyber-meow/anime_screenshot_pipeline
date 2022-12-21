import argparse
import os
import glob
import shutil
from tqdm import tqdm


def rearrange(path, dst_dir, character_list=None, copy_file=False):
    # path of the form n_faces/face_ratio.../characters
    dirname, filename = os.path.split(path)
    dirname, character_folder = os.path.split(dirname)
    face_ratio_folder = os.path.basename(dirname)
    characters = sorted(list(set(character_folder.split('+'))))
    for to_remove in ['unknown', 'ood']:
        if to_remove in characters:
            characters.remove(to_remove)
    if character_list is not None:
        for character in characters:
            assert character in character_list, \
                    f'Invalid character {character}'
    if len(characters) == 0:
        dst_dir = os.path.join(dst_dir, 'others')
    else:
        character_folder = '+'.join(characters)
        dst_dir = os.path.join(
            dst_dir, f'{len(characters)}_charcters',
            character_folder, face_ratio_folder)
    os.makedirs(dst_dir, exist_ok=True)
    new_path = os.path.join(dst_dir, filename)
    if copy_file:
        shutil.copy(path, new_path)
    else:
        shutil.move(path, new_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str,
                        help="Directory to load images")
    parser.add_argument("--dst_dir", type=str,
                        help="Directory to save images")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of move files")
    parser.add_argument(
        "--character_list", type=str, default=None,
        help="Txt file containing character names separated "
        + "by comma or new line")
    args = parser.parse_args()

    with open(args.character_list, 'r') as f:
        lines = f.readlines()
    character_list = []
    for line in lines:
        character_list.extend(line.strip().split(','))
    print(character_list)

    paths = glob.glob(f"{args.src_dir}/**/*", recursive=True)
    for path in tqdm(paths):
        if os.path.isfile(path):
            rearrange(path, args.dst_dir, character_list, args.copy)
