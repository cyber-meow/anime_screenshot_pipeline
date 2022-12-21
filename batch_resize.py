import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def resize_image(image_path, max_size):
    # Open the image
    image = Image.open(image_path)

    # Get the current width and height of the image
    width, height = image.size
    if max(width, height) <= max_size:
        return

    # Calculate the new size of the image based on the maximum size
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_width = int((max_size / height) * width)
        new_height = max_size

    # Resize the image
    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    image.save(image_path, "PNG", quality=100)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str,
                        help="Directory to load images")
    parser.add_argument("--max_image_size", type=int, default=1024)
    args = parser.parse_args()

    paths = get_files_recursively(args.src_dir)
    # Find all .png files in the current directory and its subdirectories
    for image_path in tqdm(paths):
        # Resize the image
        resize_image(image_path, args.max_image_size)
