import argparse
import os
import glob
import shutil
from tqdm import tqdm


def rearrange(path, dst_dir, character_list=None,
              max_character_number=6, copy_file=False):
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
                f'Invalid character {character} for {path}'
    if len(characters) == 0:
        character_folder = None
        dst_dir = os.path.join(dst_dir, 'others', face_ratio_folder)
    else:
        character_folder = '+'.join(characters)
        if len(characters) >= max_character_number:
            dst_dir = os.path.join(
                dst_dir, f'{max_character_number}+_charcters',
                character_folder, face_ratio_folder)
        else:
            suffix = 'character' if len(characters) == 1 else 'characters'
            dst_dir = os.path.join(
                dst_dir, f'{len(characters)}_{suffix}',
                character_folder, face_ratio_folder)
    os.makedirs(dst_dir, exist_ok=True)
    new_path = os.path.join(dst_dir, filename)
    if copy_file:
        shutil.copy(path, new_path)
    else:
        shutil.move(path, new_path)
    return character_folder, new_path


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


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str,
                        help="Directory to load images")
    parser.add_argument("--dst_dir", type=str,
                        help="Directory to save images")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of move files")
    parser.add_argument(
        "--max_character_number", type=int, default=6,
        help="If have more than X characters put X+")
    parser.add_argument(
        "--min_image_per_combination", type=int, default=1,
        help="Put others instead of character name if nnumber of images "
        + "of the character combination is smaller then this number")
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
    character_combination_dict = dict()
    for path in tqdm(paths):
        if os.path.isfile(path):
            character_folder, new_path = rearrange(
                path, args.dst_dir, character_list,
                args.max_character_number, args.copy)
            if character_folder in character_combination_dict:
                character_combination_dict[character_folder].append(new_path)
            else:
                character_combination_dict[character_folder] = [new_path]

    if args.min_image_per_combination > 1:
        merge_folder(
            character_combination_dict, args.min_image_per_combination)
        remove_empty_folders(args.dst_dir)
