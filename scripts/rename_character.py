import argparse
import os
import csv
import shutil
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


def read_class_mapping(class_mapping_csv):
    class_mapping = {}
    with open(class_mapping_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            old_class, new_class = row
            class_mapping[old_class] = new_class
    return class_mapping


def rename_folder(folder_name, class_mapping, drop_unknown_class=False):
    dirname, folder_name = os.path.split(folder_name)
    old_classes = folder_name.split('+')
    new_classes = []
    unknown_class = False
    for old_class in old_classes:
        if old_class in class_mapping:
            new_class = class_mapping[old_class]
        else:
            new_class = old_class
            if new_class not in class_mapping.values():
                unknown_class = True
        new_classes.append(new_class)
    if unknown_class and drop_unknown_class:
        return None
    return os.path.join(dirname, '+'.join(new_classes))


def modify_tags_file(tags_file, class_mapping):
    with open(tags_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith('character:'):
            old_classes = line.lstrip('character:').split(',')
            new_classes = []
            for old_class in old_classes:
                old_class = old_class.strip()
                if old_class in class_mapping:
                    new_class = class_mapping[old_class]
                else:
                    new_class = old_class
                new_classes.append(new_class)
            line = 'character: ' + ', '.join(new_classes) + '\n'
        new_lines.append(line)
    with open(tags_file, 'w') as f:
        f.writelines(new_lines)


def modify_caption_file(caption_file, class_mapping):
    with open(caption_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        for old_class in class_mapping:
            new_class = class_mapping[old_class]
            line = line.replace(old_class, new_class)
        new_lines.append(line)
    with open(caption_file, 'w') as f:
        f.writelines(new_lines)


def rename_folder_and_tags(folder, class_mapping, drop_unknown_class=False):
    new_folder_name = rename_folder(folder, class_mapping, drop_unknown_class)
    if new_folder_name is None:
        shutil.rmtree(folder)
        return
    if os.path.exists(new_folder_name):
        for file in os.listdir(folder):
            new_file_path = os.path.join(new_folder_name, file)
            os.rename(os.path.join(folder, file), new_file_path)
    else:
        os.rename(folder, new_folder_name)
    for file in get_files_recursively(new_folder_name):
        file_noext = os.path.splitext(file)[0]
        tags_file = file + '.tags'
        if os.path.exists(tags_file):
            modify_tags_file(tags_file, class_mapping)
        caption_file = file_noext + '.txt'
        if os.path.exists(caption_file):
            modify_caption_file(caption_file, class_mapping)


def get_all_subdirectories(root_dir):
    subfolders = []
    for root, dirs, files in os.walk(root_dir):
        subfolders.append(root)
    return subfolders


def main(src_dir, class_mapping_csv, drop_unknown_class):
    class_mapping = read_class_mapping(class_mapping_csv)
    for folder in tqdm(get_all_subdirectories(src_dir)):
        rename_folder_and_tags(os.path.join(
            src_dir, folder), class_mapping, drop_unknown_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True,
                        help='Path to the source directory')
    parser.add_argument('--class_mapping_csv', required=True,
                        help='Path to the class mapping CSV file')
    parser.add_argument('--drop_unknown_class', action='store_true',
                        help='Drop folders with unknown class names')
    args = parser.parse_args()
    main(args.src_dir, args.class_mapping_csv, args.drop_unknown_class)
