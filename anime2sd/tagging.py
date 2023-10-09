import json
import random


def sort_tags(tags, sort_mode):
    """
    Sorts tags based on the specified mode.

    :param tags: List or Dictionary of tags.
    :param sort_mode: Sorting mode ('original', 'shuffle', 'score').
    :return: Sorted tags.
    """
    assert sort_mode in ['original', 'shuffle', 'score']
    npeople_tags = []
    remaining_tags = []

    if 'solo' in tags:
        npeople_tags.append('solo')

    for tag in tags:
        if tag == 'solo':
            continue
        if 'girls' in tag or 'boys' in tag or tag in ['1girl', '1boy']:
            npeople_tags.append(tag)
        else:
            remaining_tags.append(tag)

    if sort_mode == 'score' and isinstance(tags, dict):
        # Sorting remaining_tags by score in descending order
        remaining_tags = sorted(
            remaining_tags,
            key=lambda tag: tags[tag],
            reverse=True  # Higher scores first
        )
    elif sort_mode == 'shuffle':
        random.shuffle(remaining_tags)

    return npeople_tags + remaining_tags


def parse_overlap_tags(json_file):
    """
    Parses a JSON file to extract the 'query' and 'has_overlap' fields.

    :param json_file: Path to the JSON file.
    :return: A dictionary with 'query' as keys and 'has_overlap' as values.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    overlap_tags_dict = {
        entry['query']: entry['has_overlap']
        for entry in data if 'has_overlap' in entry and entry['has_overlap']
    }

    return overlap_tags_dict


def remove_blacklisted_tags(tags, blacklisted_tags):
    """
    Remove blacklisted tags from the list or dictionary of tags.

    :param tags: List or dictionary of tags.
    :param blacklisted_tags: Set of blacklisted tags.
    :return: List or dictionary of tags after removing blacklisted tags.
    """
    # Handle both underscore and whitespace in tags
    blacklist = set(tag.replace(' ', '_') for tag in blacklisted_tags)
    blacklist_update = set(tag.replace('_', ' ') for tag in blacklist)
    blacklist.update(blacklist_update)

    if isinstance(tags, dict):
        return {tag: value for tag, value in tags.items()
                if tag not in blacklisted_tags and
                tag.replace(' ', '_') not in blacklisted_tags and
                tag.replace('_', ' ') not in blacklisted_tags}
    elif isinstance(tags, list):
        return [tag for tag in tags
                if tag not in blacklisted_tags and
                tag.replace(' ', '_') not in blacklisted_tags and
                tag.replace('_', ' ') not in blacklisted_tags]
    else:
        raise ValueError(f"Unsuppored types {type(tags)} for {tags}")


def is_basic_character_tag(tag):
    """
    Check if the tag is a basic character tag.

    :param tag: A single tag.
    :return: Boolean value indicating whether the tag is a basic character tag.
    """
    whitelist = set([
        'drill',
        'pubic_hair',
        'closed_eyes',
        'half-closed_eyes',
        'empty_eyes',
    ])
    whitelist_update = set(tag.replace('_', ' ') for tag in whitelist)
    whitelist.update(whitelist_update)
    if tag in whitelist:
        return False

    suffixes = [
        'eyes', 'skin', 'hair', 'bun', 'bangs', 'cut', 'sidelocks',
        'twintails', 'braid', 'braids', 'afro', 'ahoge', 'drill',
        'drills', 'bald', 'dreadlocks', 'side up', 'ponytail', 'updo',
        'beard', 'mustache', 'pointy ears',
    ]
    prefixes = ['hair over', 'hair between']

    tag = tag.replace('_', ' ')
    result_suffix = any(tag.endswith(suffix) for suffix in suffixes)
    result_prefix = any(tag.startswith(prefix) for prefix in prefixes)
    return result_suffix or result_prefix


def remove_basic_character_tags(tags):
    """
    Remove basic character tags from the list or dictionary of tags.

    :param tags: List or dictionary of tags.
    :return: List or dictionary of tags after removing basic character tags.
    """
    if isinstance(tags, dict):
        return {
            tag: value for tag, value in tags.items()
            if not is_basic_character_tag(tag)
        }
    elif isinstance(tags, list):
        return [tag for tag in tags if not is_basic_character_tag(tag)]
    else:
        raise ValueError(f"Unsuppored types {type(tags)} for {tags}")


def remove_overlap_tags(tags, overlap_tags_dict):
    """
    Removes overlap tags from the list of tags.

    :param tags: List or Dictionary of tags.
    :param overlap_tags_dict: Dictionary with overlap tag information.
        Assume here to take the underscore format
    :return: A list or dictionary with overlap tags removed.
    """
    # If tags is a dictionary, extract the keys for processing
    # and remember to return a dictionary
    return_as_dict = False
    original_tags = tags
    if isinstance(tags, dict):
        return_as_dict = True
        tags = list(tags.keys())

    result_tags = []
    tags_underscore = [tag.replace(' ', '_') for tag in tags]

    for tag, tag_ in zip(tags, tags_underscore):

        to_remove = False

        # Case 1: If the tag is a key and some of
        # the associated values are in tags
        if tag_ in overlap_tags_dict:
            overlap_values = set(
                val for val in overlap_tags_dict[tag_])
            if overlap_values.intersection(set(tags_underscore)):
                to_remove = True

        # Checking superword condition separately
        for tag_another in tags:
            if tag in tag_another and tag != tag_another:
                to_remove = True
                break

        if not to_remove:
            result_tags.append(tag)

    # If the input was a dictionary
    # return as a dictionary with the same values
    if return_as_dict:
        result_tags = {tag: original_tags[tag] for tag in result_tags}

    return result_tags
