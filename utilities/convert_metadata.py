import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def get_images_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]',
        '*.[Jj][Pp][Gg]',
        '*.[Jj][Pp][Ee][Gg]',
        '*.[Ww][Ee][Bb][Pp]',
        '*.[Gg][Ii][Ff]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def convert_metadata(src_dir):
    img_paths = get_images_recursively(src_dir)
    for img_path in tqdm(img_paths):
        meta_file_path = os.path.splitext(img_path)[0] + '.json'
        if os.path.exists(meta_file_path):
            # Rename the metadata file
            new_meta_file_path = os.path.join(
                os.path.dirname(img_path),
                f".{os.path.splitext(os.path.basename(img_path))[0]}_meta.json"
            )
            os.rename(meta_file_path, new_meta_file_path)

            # Modify its content
            with open(new_meta_file_path, 'r') as meta_file:
                meta_data = json.load(meta_file)

            # Rename fields
            if "character" in meta_data:
                meta_data["characters"] = meta_data.pop("character")
            if "general" in meta_data:
                meta_data["type"] = meta_data.pop("general")

            # Add new fields
            meta_data["path"] = os.path.abspath(img_path)
            meta_data["current_path"] = os.path.abspath(img_path)
            meta_data["filename"] = os.path.basename(img_path)

            # Save the modified metadata
            with open(new_meta_file_path, 'w') as meta_file:
                json.dump(meta_data, meta_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert metadata files.")
    parser.add_argument("--src_dir", required=True,
                        help="Directory containing the metadata to modify.")
    args = parser.parse_args()
    convert_metadata(args.src_dir)
