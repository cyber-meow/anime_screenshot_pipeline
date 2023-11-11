import os
import re
import json
import logging
from tqdm import tqdm
from typing import List, Dict

from anime2sd.basics import get_images_recursively, get_corr_meta_names
from anime2sd.tagging_basics import get_all_singular_plural_forms


def _split_to_words(text: str):
    """
    Split a string into words separated by whitespaces and underscores

    :param text: The input text to split.
    :type text: str
    :return: List of lowercase words.
    :rtype: List[str]
    """
    words = [subtext.split("_") for subtext in text.split()]
    # return [word.lower() for word in re.split(r'[\W_]+', text) if word]
    return [word.lower() for sublist in words for word in sublist if word]


def _match_suffix(tag: str, suffix: str):
    """
    Check if a tag matches a given suffix.

    :param tag: The tag to check.
    :type tag: str
    :param suffix: The suffix to match.
    :type suffix: str
    :return: True if the tag matches the suffix, False otherwise.
    :rtype: bool
    """
    tag_words = _split_to_words(tag)
    suffix_words = _split_to_words(suffix)

    return tag_words[-len(suffix_words) :] == suffix_words


def _match_prefix(tag: str, prefix: str):
    """
    Check if a tag matches a given prefix.

    :param tag: The tag to check.
    :type tag: str
    :param prefix: The prefix to match.
    :type prefix: str
    :return: True if the tag matches the prefix, False otherwise.
    :rtype: bool
    """
    tag_words = _split_to_words(tag)
    prefix_words = _split_to_words(prefix)
    return tag_words[: len(prefix_words)] == prefix_words


class CharacterTagProcessor(object):
    _CHAR_WHITELIST = [
        "drill",
        "pubic hair",
        "closed eyes",
        "half-closed eyes",
        "empty eyes",
        "fake tail",
    ]

    _MAX_DIFFICULTY = 2

    _CHAR_SUFFIXES = [
        # Pure human
        [
            "man",
            "woman",
            "eyes",
            "skin",
            "hair",
            "bun",
            "bangs",
            "cut",
            "sidelocks",
            "twintails",
            "braid",
            "braids",
            "afro",
            "ahoge",
            "drill",
            "bald",
            "dreadlocks",
            "side up",
            "ponytail",
            "updo",
            "beard",
            "mustache",
            "goatee",
            "hair intake",
            "thick eyebrows",
            "otoko no ko",
            "bishounen",
            "short hair with long locks",
            "one eye covered",
        ],
        # Furry, Mecha etc
        [
            "fang",
            "mark",
            "freckles",
            "ear",
            "horn",
            "fur",
            "halo",
            "wings",
            "heterochromia",
            "tail",
            "animal ear fluff",
            "girl",
            "boy",
        ],
    ]

    _CHAR_PREFIXES = [
        ["hair over", "hair between", "dark-skinned", "mature", "old"],
        ["mole", "scar", "furry", "muscular"],
    ]

    def __init__(
        self,
        drop_difficulty: int = 2,
        emb_min_difficulty: int = 1,
        emb_max_difficutly: int = 2,
    ):
        """
        Generates default character tag lists based on the specified
        difficulties.

        This function returns lists of tags that should be whitelisted,
        dropped, or used for embedding initialization based on the provided
        difficulty levels.

        Args:
            drop_difficulty (int):
                The difficulty level up to which tags should be dropped. Tags with
                difficulty less than this value will be added to the drop lists.
                Defaults to 2.
            emb_min_difficulty:
                The difficulty level from which tags should be used for embedding
                initialization.
                Defaults to 1.
            emb_max_difficutly (int, optional):
                The difficulty level up to which tags should be used for embedding
                initialization.
                Defaults to 2.

        Initialize:
            tuple: A tuple containing five lists:
                - whitelist: List of tags that are always whitelisted.
                - drop_prefixes: List of tag prefixes to be dropped.
                - drop_suffixes: List of tag suffixes to be dropped.
                - emb_init_prefixes: List of tag prefixes for embedding
                initialization.
                - emb_init_suffixes: List of tag suffixes for embedding
                initialization.

        Note:
            The function uses predefined constants `_CHAR_WHITELIST`,
            `_CHAR_PREFIXES`, `_CHAR_SUFFIXES`, and `_MAX_DIFFICULTY` to
            determine the tags based on the difficulty levels.
        """
        self.whitelist = get_all_singular_plural_forms(self._CHAR_WHITELIST)
        self.drop_prefixes = []
        self.drop_suffixes = []
        self.emb_init_prefixes = []
        self.emb_init_suffixes = []
        for difficulty in range(0, min(drop_difficulty, self._MAX_DIFFICULTY)):
            self.drop_prefixes.extend(self._CHAR_PREFIXES[difficulty])
            self.drop_suffixes.extend(self._CHAR_SUFFIXES[difficulty])
        for difficulty in range(
            emb_min_difficulty, min(emb_max_difficutly, self._MAX_DIFFICULTY)
        ):
            self.emb_init_prefixes.extend(self._CHAR_PREFIXES[difficulty])
            self.emb_init_suffixes.extend(self._CHAR_SUFFIXES[difficulty])
        self.drop_suffixes = get_all_singular_plural_forms(self.drop_suffixes)
        self.emb_init_suffixes = get_all_singular_plural_forms(self.emb_init_suffixes)

    def is_character_tag(self, tag, mode="drop"):
        """
        Check if a tag is a basic character tag
        by matching with predefined whitelisted and blacklisted patterns.

        :param tag: The tag to check.
        :type tag: str
        :return: True if the tag is a basic character tag, False otherwise.
        :rtype: bool
        """
        assert mode in ["drop", "emb_init"]
        if tag in self.whitelist:
            return False
        else:
            if mode == "drop":
                suffixes = self.drop_suffixes
                prefixes = self.drop_prefixes
            else:
                suffixes = self.emb_init_suffixes
                prefixes = self.emb_init_prefixes
            return any(_match_suffix(tag, suffix) for suffix in suffixes) or any(
                _match_prefix(tag, prefix) for prefix in prefixes
            )

    def drop_character_tags(self, tags):
        """
        Drop basic character tags from the given list or mapping of tags.

        :param tags: List or mapping of tags to be filtered.
        :type tags: Union[List[str], Mapping[str, float]]
        :return: Filtered list or mapping of tags without the character tags.
        :rtype: Union[List[str], Mapping[str, float]]
        :raises TypeError: If the input tags are neither list nor dictionary.
        """
        if isinstance(tags, dict):
            return {
                tag: value
                for tag, value in tags.items()
                if not self.is_character_tag(tag, "drop")
            }
        elif isinstance(tags, list):
            return [tag for tag in tags if not self.is_character_tag(tag, "drop")]
        else:
            raise TypeError(
                "Unsupported types of tags, dict or list expected, "
                f"but {tags!r} found."
            )

    def categorize_tags(self, tags):
        """
        Categorize tags into 'kept', 'dropped', and 'emb_init'.

        Args:
            tags: List of tags to be categorized.

        Returns:
            Dictionary containing the categorized tags.
        """
        kept = []
        dropped = []
        emb_init = []

        for tag in tags:
            if self.is_character_tag(tag, mode="drop"):
                dropped.append(tag)
            else:
                kept.append(tag)
            if self.is_character_tag(tag, mode="emb_init"):
                emb_init.append(tag)

        return {"kept": kept, "dropped": dropped, "emb_init": emb_init}

    def categorize_character_tag_dict(self, character_tag_dict: Dict[str, List[str]]):
        """
        Update character_tag_dict by grouping characters based on shared embeddings
        and categorizing tags for these embeddings.

        Args:
            character_tag_dict: Dictionary of characters and their core tags.

        Returns:
            Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
                - The first dictionary contains categorized tags at the
                  character form level.
                - The second dictionary contains 'emb_init' tags at the embedding level.
        """
        # Categorize tags for each character form initially
        categorized_character_tags = {
            character: self.categorize_tags(tags)
            for character, tags in character_tag_dict.items()
        }

        # Group characters by embedding
        embedding_dict = {}
        for character, categorized_tags in categorized_character_tags.items():
            embedding = character.split()[0]
            if embedding not in embedding_dict:
                embedding_dict[embedding] = {"characters": [], "tags": []}
            embedding_dict[embedding]["characters"].append(character)
            embedding_dict[embedding]["tags"].append(categorized_tags["emb_init"])

        # Find intersection of emb_init tags for each embedding
        character_form_dict = {}
        embedding_level_dict = {}
        for embedding, data in embedding_dict.items():
            shared_emb_init = set(data["tags"][0])
            for tags in data["tags"][1:]:
                shared_emb_init.intersection_update(tags)
            shared_emb_init = list(shared_emb_init)

            # Update character form level dictionary
            for character in data["characters"]:
                character_form_dict[character] = categorized_character_tags[character]
                # Update emb_init for each character form
                categorized_character_tags[character]["emb_init"] = shared_emb_init
                for tag in shared_emb_init:
                    if tag in character_form_dict[character]["kept"]:
                        character_form_dict[character]["kept"].remove(tag)

            # Save the shared emb_init for the embedding
            embedding_level_dict[embedding] = shared_emb_init

        return character_form_dict, embedding_level_dict


"""
For character core tags
"""

# close-up not working
_BLACKLISTED_WORDS_CORE = [
    "solo",
    "1girl",
    "1boy",
    "2girls",
    "2boys",
    "3girls",
    "3boys",
    "girls",
    "boys",
    "body",
    "background",
    "quality",
    "chibi",
    "monochrome",
    "comic",
    "looking",
    "text",
    "signature",
    "peeking",
    "focus",
    "smile",
    "mouth",
    "anime",
    "screenshot",
    "sky",
    "tree",
    "cloud",
    "day",
    "indoors",
    "outdoors",
    "close-up",
    "window",
    "curtains",
]


# TODO: Improved the blacklist mechanism here (maybe use whitelist instead)
# for example mole under mouth should be retained
def contains_blacklisted_word_core(tag: str):
    words = [word for word in re.split(r"[\W_]+", tag.lower()) if word]
    return any((word in _BLACKLISTED_WORDS_CORE) for word in words)


def update_character_tag_dict(character, tags, character_tag_dict):
    """
    Update the tag dictionary for a given character.

    Args:
        character (str): The character's name.
        tags (list): A list of tags associated with the character.
        character_tag_dict (dict): The dictionary to update.

    Returns:
        None
    """
    if character not in character_tag_dict:
        character_tag_dict[character] = [0, dict()]
    for tag in tags:
        if tag not in character_tag_dict[character][1]:
            character_tag_dict[character][1][tag] = 1
        else:
            character_tag_dict[character][1][tag] += 1
    character_tag_dict[character][0] += 1


def get_frequent_tags(character_tag_dict, frequency_threshold):
    """
    Extract tags for each character that appear more frequently
    than the given threshold.

    Args:
        character_tag_dict (dict): The character-tag dictionary.
        frequency_threshold (int): The minimum frequency for a tag
        to be considered frequent.

    Returns:
        dict: A dictionary where keys are characters and values are lists
        of frequent tags.
    """
    frequent_tags = {}
    for character, (total, tags) in character_tag_dict.items():
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        frequent_tags[character] = [
            tag
            for tag, count in sorted_tags
            if count >= frequency_threshold * total
            and not contains_blacklisted_word_core(tag)
        ]
    return frequent_tags


def get_character_core_tags(folder_path, frequency_threshold):
    """
    For each character in the given folder, find the tags whose appearance
    frequency is higher than a certain threshold.
    A character is considered only if there contains solo images of the character.

    Args:
        folder_path (str): The path to the folder containing image metadata.
        frequency_threshold (float): The minimum frequency for a tag to be
        considered frequent.

    Returns:
        dict: A dictionary where keys are characters and values are lists
        of frequent tags.
    """
    img_paths = get_images_recursively(folder_path)
    character_tag_dict = dict()
    logging.info("Search for core tags...")
    for img_path in tqdm(img_paths):
        meta_file_path, _ = get_corr_meta_names(img_path)
        if not os.path.exists(meta_file_path):
            continue
        with open(meta_file_path, "r") as meta_file:
            meta_data = json.load(meta_file)
        if "characters" not in meta_data:
            continue
        characters = meta_data["characters"]
        if "processed_tags" in meta_data:
            tags = meta_data["processed_tags"]
        elif "tags" in meta_data:
            tags = meta_data["tags"]
        else:
            continue
        # Only update for single character image
        if len(characters) == 1:
            update_character_tag_dict(characters[0], tags, character_tag_dict)
    frequent_tag_dict = get_frequent_tags(character_tag_dict, frequency_threshold)
    return frequent_tag_dict


def save_core_tag_info(data, json_output, wildcard_ouput):
    with open(json_output, "w") as f:
        json.dump(data, f, indent=4)
    with open(wildcard_ouput, "w") as f:
        for character, tags in data.items():
            f.write(character + "\n")
            # The case where we split into kept, dropped, and emb_init
            if isinstance(tags, dict):
                tags = tags["kept"]
            if len(tags) > 0:
                f.write(character + ", " + ", ".join(tags) + "\n")


def get_character_core_tags_and_save(
    folder_path, core_tag_output, wildcard_output, frequency_threshold
):
    frequent_tags_dict = get_character_core_tags(folder_path, frequency_threshold)
    save_core_tag_info(frequent_tags_dict, core_tag_output, wildcard_output)
    return frequent_tags_dict


def drop_character_core_tags(characters, tags, character_tag_dict):
    """
    Drops tags from a list based on the provided character_tag_dict.

    :param characters: List of characters.
    :type characters: List[str]
    :param tags: List of tags to be filtered.
    :type tags: List[str]
    :param character_tag_dict: Dictionary containing tags for characters.
    :type character_tag_dict: dict
    :return: Filtered list of tags.
    :rtype: List[str]
    """
    filtered_tags = tags.copy()
    to_drop = []
    for character in characters:
        if character not in character_tag_dict:
            logging.warning(
                f"Character '{character}' not found in core tag dictionary."
            )
            continue

        # Get the tags to be dropped for this character
        to_drop.extend(
            character_tag_dict[character].get("dropped", [])
            + character_tag_dict[character].get("emb_init", [])
        )

    # Remove the tags from the filtered_tags list
    filtered_tags = [tag for tag in filtered_tags if tag not in to_drop]

    return filtered_tags
