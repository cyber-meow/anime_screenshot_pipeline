import argparse
import os
import shutil
import json
from tqdm import tqdm

from anime2sd.basics import get_images_recursively
from anime2sd.basics import get_corr_meta_names


def replace_tag_in_file(file_path, old_name, new_name, separator):
    with open(file_path, "r") as f:
        content = f.read()
    # Split the content by commas to get the tags
    tags = content.split(separator)
    # Replace the old_name tag with new_name if it exists
    tags = [new_name if tag.strip() == old_name else tag for tag in tags]
    # Join the tags back with commas
    content = separator.join(tags)
    with open(file_path, "w") as f:
        f.write(content)


def replace_character_in_json(json_path, old_name, new_name):
    with open(json_path, "r") as f:
        content = json.load(f)
    if "characters" in content:
        content["characters"] = [
            new_name if c == old_name else c for c in content["characters"]
        ]
    with open(json_path, "w") as f:
        json.dump(content, f, indent=4)


def rename_folders(src_dir, old_name, new_name):
    for root, dirs, _ in os.walk(src_dir):
        for dir_name in dirs:
            name_parts = dir_name.split("+")
            if old_name in name_parts:
                # Replace only the exact old_name
                name_parts = [
                    new_name if part == old_name else part for part in name_parts
                ]
                new_dir_name = "+".join(name_parts)
                shutil.move(
                    os.path.join(root, dir_name), os.path.join(root, new_dir_name)
                )


def main():
    parser = argparse.ArgumentParser(description="Replace tags in txt files.")
    parser.add_argument(
        "--src_dir", required=True, help="Source directory containing txt files."
    )
    parser.add_argument(
        "--old_name", required=True, help="Old character name to be replaced."
    )
    parser.add_argument(
        "--new_name", required=True, help="New character name to replace the old one."
    )
    parser.add_argument(
        "--caption_separation",
        type=str,
        default=",",
        help="Symbol used to separate character names in caption",
    )

    args = parser.parse_args()

    # Walk through the src_dir and find all txt files
    for img_path in tqdm(get_images_recursively(args.src_dir)):
        img_noext, _ = os.path.splitext(img_path)
        for potential_exts in [".txt", ".characters"]:
            potential_file = img_noext + potential_exts
            if os.path.exists(potential_file):
                replace_tag_in_file(
                    potential_file,
                    args.old_name,
                    args.new_name,
                    args.caption_separation,
                )
        meta_path, _ = get_corr_meta_names(img_path)
        if os.path.exists(meta_path):
            replace_character_in_json(meta_path, args.old_name, args.new_name)
    rename_folders(args.src_dir, args.old_name, args.new_name)


if __name__ == "__main__":
    main()
