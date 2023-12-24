import os
import json
import toml
import argparse

from tqdm import tqdm
from pathlib import Path


def get_files_recursively(folder_path):
    allowed_patterns = [
        "*.[Pp][Nn][Gg]",
        "*.[Jj][Pp][Gg]",
        "*.[Jj][Pp][Ee][Gg]",
        "*.[Gg][Ii][Ff]",
        "*.[Ww][Ee][Bb][Pp]",
    ]

    image_path_list = [
        str(path)
        for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def get_basic_metadata(file_path, args):
    filename_noext = os.path.splitext(file_path)[0]
    json_file = filename_noext + ".json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = dict()

    if args.general_description is not None:
        metadata["general"] = args.general_description
    if args.retrieve_description_from_folder:
        if "general" not in metadata:
            metadata["general"] = ""
        to_prepend = os.path.basename(os.path.dirname(file_path))
        metadata["general"] += " " + "".join(to_prepend.replace("_", " "))

    return metadata


def to_list(line):
    items = [item.strip() for item in line.split(",")]
    return items


def retrieve_tag_info(
    basic_info, tags_content, tag_extension, keep_charcacter, remove_before_girl
):
    if tag_extension == "toml":
        for key in tags_content:
            info = tags_content[key]
            if key == "character":
                if not keep_charcacter:
                    characters = to_list(info)
                    for to_remove in ["unknown", "ood"]:
                        if to_remove in characters:
                            characters.remove(to_remove)
                    basic_info["character"] = characters
            elif key in ["copyright", "artist", "tags"]:
                basic_info[key] = to_list(info)
            else:
                basic_info[key] = info

    else:
        for line in tags_content:
            # If tags contain character information use it instead
            if line.startswith("character: "):
                if not keep_charcacter:
                    characters = to_list(line.lstrip("character:"))
                    for to_remove in ["unknown", "ood"]:
                        if to_remove in characters:
                            characters.remove(to_remove)
                    basic_info["character"] = characters
            elif line.startswith("copyright: "):
                basic_info["copyright"] = to_list(line.lstrip("copyright:"))
            elif line.startswith("artist: "):
                basic_info["artist"] = to_list(line.lstrip("artist:").strip())
            elif line.startswith("general: "):
                basic_info["tags"] = to_list(line.lstrip("general: "))
            elif line.startswith("rating: "):
                basic_info["rating"] = line.lstrip("rating:").strip()
            # This is the case when there is a single line of tags
            elif line.startswith("score: "):
                basic_info["score"] = line.lstrip("score:").strip()
            else:
                basic_info["tags"] = to_list(line.strip())

    if remove_before_girl:
        tags = basic_info["tags"]
        glist = ["1girl", "2girls", "3girls", "4girls", "5girls", "6+girls"]
        for k, tag in enumerate(tags):
            if tag in glist:
                del tags[:k]
                break
    return basic_info


def get_npeople_from_tags(tags):
    girl_dictinary = {
        "1girl": 1,
        "6+girls": 6,
    }
    boy_dictinary = {
        "1boy": 1,
        "6+boys": 6,
    }
    for k in range(2, 6):
        girl_dictinary[f"{k}girls"] = k
        boy_dictinary[f"{k}boys"] = k

    n_girls = 0
    n_boys = 0
    for key in girl_dictinary:
        if key in tags:
            n_girls = max(n_girls, girl_dictinary[key])
    for key in boy_dictinary:
        if key in tags:
            n_boys = max(n_boys, boy_dictinary[key])
    n_people = n_girls + n_boys
    if n_people >= 6:
        n_people = "many"
    return n_people


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory to load images")
    parser.add_argument(
        "--general_description", default=None, help="General description of the files"
    )
    parser.add_argument("--retrieve_description_from_folder", action="store_true")
    parser.add_argument(
        "--use_tags", action="store_true", help="Use tag files to retrieve information"
    )
    parser.add_argument(
        "--tag_extension",
        choices=["tags", "txt", "toml"],
        type=str,
        help="Extension of tag files",
    )
    parser.add_argument(
        "--remove_before_girl",
        action="store_true",
        help="Remove the tags that appear before [k]girl(s)",
    )
    parser.add_argument(
        "--keep_character",
        action="store_true",
        help="Do not overwrite character information",
    )
    args = parser.parse_args()

    file_paths = get_files_recursively(args.src_dir)
    for file_path in tqdm(file_paths):
        metadata = get_basic_metadata(file_path, args)
        if args.use_tags:
            tags_file = file_path + "." + args.tag_extension
            if os.path.exists(tags_file):
                with open(tags_file, "r") as f:
                    if args.tag_extension == "toml":
                        tags_data = toml.load(f)["meta"]
                    else:
                        tags_data = f.readlines()
                metadata = retrieve_tag_info(
                    metadata,
                    tags_data,
                    args.tag_extension,
                    args.keep_character,
                    args.remove_before_girl,
                )
                if "tags" in metadata:
                    n_people = get_npeople_from_tags(metadata["tags"])
                    metadata["n_people"] = n_people
            else:
                print(f"Warning: tags file {tags_file} not found")
        filename_noext = os.path.splitext(file_path)[0]
        savefile = filename_noext + ".json"
        with open(savefile, "w") as f:
            json.dump(metadata, f)
