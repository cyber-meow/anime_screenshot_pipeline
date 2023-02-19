import os
import argparse


def list_image_subfolders(path):
    """Recursively list all subfolders of a directory."""
    subfolders = []
    contain_subfolder = False
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            contain_subfolder = True
            subfolders.extend(list_image_subfolders(item_path))
    if not contain_subfolder:
        subfolders.append(path)
    return subfolders


def get_new_path(src_dir, path):
    multiply_file = os.path.join(path, 'multiply.txt')
    repeat = 1
    if os.path.exists(multiply_file):
        with open(multiply_file, 'r') as f:
            repeat = int(float(f.readline().strip()))
    subpath = path.replace(src_dir, '').lstrip('/')
    new_subpath = subpath.replace('/', '-')
    return os.path.join(src_dir, f'{repeat}_{new_subpath}')


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True,
                        help='Path to the source directory')
    args = parser.parse_args()
    for path in list_image_subfolders(args.src_dir):
        new_path = get_new_path(args.src_dir, path)
        os.rename(path, new_path)
    remove_empty_folders(args.src_dir)
