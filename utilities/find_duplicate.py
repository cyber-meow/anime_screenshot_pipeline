import os
import sys


def find_duplicates(dir1, dir2):

    duplicates = []

    # Get the list of files in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Iterate over the files in the first directory
    for file in files1:
        # Check if the file also exists in the second directory
        if file in files2:
            duplicates.append(file)
    return duplicates, files1, files2


def move_to_subfolder(file, dir1, dir2,
                      files1=None, files2=None,
                      subfolder='duplicate'):

    file1_path = os.path.join(dir1, file)
    file2_path = os.path.join(dir2, file)

    duplicate_path1 = os.path.join(dir1, subfolder)
    duplicate_path2 = os.path.join(dir2, subfolder)
    if not os.path.exists(duplicate_path1):
        os.makedirs(duplicate_path1)
    if not os.path.exists(duplicate_path2):
        os.makedirs(duplicate_path2)
    os.rename(file1_path, os.path.join(duplicate_path1, file))
    os.rename(file2_path, os.path.join(duplicate_path2, file))

    if files1 is None:
        files1 = os.listdir(dir1)
    if files2 is None:
        files2 = os.listdir(dir2)

    txt_file = f"{file}.txt"
    if txt_file in files1:
        os.rename(os.path.join(dir1, txt_file),
                  os.path.join(duplicate_path1, txt_file))
    if txt_file in files2:
        os.rename(os.path.join(dir2, txt_file),
                  os.path.join(duplicate_path2, txt_file))

    txt_file = f"{file}.tags"
    if txt_file in files1:
        os.rename(os.path.join(dir1, txt_file),
                  os.path.join(duplicate_path1, txt_file))
    if txt_file in files2:
        os.rename(os.path.join(dir2, txt_file),
                  os.path.join(duplicate_path2, txt_file))


if __name__ == '__main__':

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    duplicates, files1, files2 = find_duplicates(dir1, dir2)
    print(duplicates)
    image_extensions = ['.png', '.jpg', '.gif', '.jpeg']
    for file_path in duplicates:
        if os.path.splitext(file_path)[1].lower() in image_extensions:
            move_to_subfolder(file_path, dir1, dir2, files1, files2)
