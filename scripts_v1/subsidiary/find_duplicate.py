import os
import sys
from pathlib import Path
from tqdm import tqdm


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


def find_duplicates(dir1, dir2):

    duplicates1 = []
    duplicates2 = []
    detected_files = []

    # Get the list of files in each directory
    paths1 = get_files_recursively(dir1)
    files1 = [os.path.basename(path) for path in paths1]
    paths2 = get_files_recursively(dir2)
    files2 = [os.path.basename(path) for path in paths2]

    # Iterate over the files in the first directory
    for path1, file1 in zip(paths1, files1):
        # Check if the file also exists in the second directory
        for path2, file2 in zip(paths2, files2):
            if file1 == file2:
                duplicates1.append(path1)
                # Some duplicates in dir2 may remain
                if file1 not in detected_files:
                    duplicates2.append(path2)
                break
    return duplicates1, duplicates2, paths1, paths2


def move_to_subfolder(path1, path2, dir1, dir2,
                      files1=None, files2=None,
                      subfolder='duplicate'):

    file1_path = path1
    file2_path = path2

    duplicate_path1 = os.path.join(dir1, subfolder)
    duplicate_path2 = os.path.join(dir2, subfolder)
    if not os.path.exists(duplicate_path1):
        os.makedirs(duplicate_path1)
    if not os.path.exists(duplicate_path2):
        os.makedirs(duplicate_path2)
    file = os.path.basename(file1_path)
    os.rename(file1_path, os.path.join(duplicate_path1, file))
    # os.rename(file2_path, os.path.join(duplicate_path2, file))

    if files1 is None:
        files1 = os.listdir(dir1)
    if files2 is None:
        files2 = os.listdir(dir2)

    txt_file = f"{file1_path}.txt"
    if os.path.exists(txt_file):
        os.rename(txt_file,
                  os.path.join(duplicate_path1, os.path.basename(txt_file)))
    # txt_file = f"{file2_path}.txt"
    # if txt_file in files2:
    #     os.rename(os.path.join(dir2, txt_file),
    #               os.path.join(duplicate_path2, os.path.basename(txt_file)))

    txt_file = f"{file1_path}.tags"
    if os.path.exists(txt_file):
        os.rename(txt_file,
                  os.path.join(duplicate_path1, os.path.basename(txt_file)))
    # txt_file = f"{file2_path}.tags"
    # if txt_file in files2:
    #     os.rename(os.path.join(dir2, txt_file),
    #               os.path.join(duplicate_path2, os.path.basename(txt_file)))


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


if __name__ == '__main__':

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    duplicates1, duplicates2, paths1, paths2 = find_duplicates(dir1, dir2)
    print(len(duplicates1))
    print(len(duplicates2))
    image_extensions = ['.png', '.jpg', '.gif', '.jpeg', '.webp']
    for file_path1, file_path2 in tqdm(zip(duplicates1, duplicates2)):
        if os.path.splitext(file_path1)[1].lower() in image_extensions:
            move_to_subfolder(
                file_path1, file_path2, dir1, dir2, paths1, paths2)
    remove_empty_folders(dir1)
