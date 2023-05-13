import os
import shutil
import argparse
import hashlib
from pathlib import Path
from tqdm.auto import tqdm


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]',
        '*.[Jj][Pp][Gg]',
        '*.[Jj][Pp][Ee][Gg]',
        '*.[Gg][Ii][Ff]'
        '*.[Ww][Ee][Bb][Pp]'
    ]

    image_path_list = [
        str(path)
        for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)]

    return image_path_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='.')
    parser.add_argument('--dest', type=str, default='files_md5')
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    for path in tqdm(get_files_recursively(args.source)):
        # Open,close, read file and calculate MD5 on its contents
        with open(path, 'rb') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            md5_returned = hashlib.md5(data).hexdigest()
            _, ext = os.path.splitext(path)
            new_path = os.path.join(args.dest, md5_returned + ext)
            shutil.copy(path, new_path)
