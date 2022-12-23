import argparse
import os
import json
import random

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


def dict_to_caption(info_dict, args):
    caption = ""
    if random.random() < args.use_count_prob and 'count' in info_dict:
        count = info_dict['count']
        suffix = args.count_singular if count == 1 else args.count_plural
        caption += f'{count}{suffix}'
    if (random.random() < args.use_character_prob
            and 'characters' in info_dict):
        characters = info_dict['characters']
        if len(characters) > 0:
            caption += ', ' + ' '.join(characters)
    if random.random() < args.use_copyright_prob and 'copyright' in info_dict:
        copyright = info_dict['copyright']
        if len(copyright) > 0:
            caption += ', from ' + ' '.join(copyright)
    if random.random() < args.use_general_prob and 'general' in info_dict:
        caption += ', ' + info_dict['general']
    if random.random() < args.use_artist_prob and 'artist' in info_dict:
        artist = info_dict['artist']
        if len(artist) > 0:
            caption += ', style of ' + ' '.join(artist)
    if random.random() < args.use_facepos_prob and 'facepos' in info_dict:
        facepos = info_dict['facepos']
        if len(facepos) > 0:
            caption += ', ' + ' '.join(facepos)
    if random.random() < args.use_tags_prob and 'tags' in info_dict:
        tags = info_dict['tags']
        if args.shuffle_tags:
            random.shuffle(tags)
        tags = tags[:args.max_tag_number]
        if len(tags) > 0:
            caption += ', ' + ', '.join(tags)
    return caption


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory to load images")
    parser.add_argument('--use_count_prob', type=float, default=1)
    parser.add_argument('--count_singular', type=str, default='person')
    parser.add_argument('--count_plural', type=str, default='people')
    parser.add_argument('--use_character_prob', type=float, default=1)
    parser.add_argument('--use_general_prob', type=float, default=1)
    parser.add_argument('--use_copyright_prob', type=float, default=0)
    parser.add_argument('--use_artist_prob', type=float, default=1)
    parser.add_argument('--use_facepos_prob', type=float, default=1)
    parser.add_argument('--use_tags_prob', type=float, default=1)
    parser.add_argument('--max_tag_number', type=int, default=15)
    parser.add_argument('--shuffle_tags', action='store_true')
    args = parser.parse_args()

    files = get_files_recursively(args.src_dir)
    for file in tqdm(files):
        filename_noext = os.path.splitext(file)[0]
        with open(filename_noext + '.json', 'r') as f:
            info_dict = json.load(f)
        caption = dict_to_caption(info_dict, args)
        with open(filename_noext + '.txt', 'w') as f:
            f.write(caption)
