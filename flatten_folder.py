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


def get_new_path(src_dir, path, separator):
    multiply_file = os.path.join(path, 'multiply.txt')
    repeat = 1
    if os.path.exists(multiply_file):
        with open(multiply_file, 'r') as f:
            repeat = round(float(f.readline().strip()))
    subpath = path.replace(src_dir, '').lstrip(os.path.sep)
    new_subpath = subpath.replace(os.path.sep, separator)
    return os.path.join(src_dir, f'{repeat}_{new_subpath}')


def revert_path(src_dir, path, separator):
    subpath = '_'.join(
        path.replace(src_dir, '').lstrip(os.path.sep).split('_')[1:])
    new_subpath = subpath.replace(separator, os.path.sep)
    return os.path.join(src_dir, new_subpath)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True,
                        help='Path to the source directory')
    parser.add_argument('--separator', default='~',
                        help='String to sepearte folders of different levels')
    parser.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    for path in list_image_subfolders(args.src_dir):
        if args.revert:
            new_path = revert_path(args.src_dir, path, args.separator)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
        else:
            new_path = get_new_path(args.src_dir, path, args.separator)
        os.rename(path, new_path)
    remove_empty_folders(args.src_dir)
