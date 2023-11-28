import os
import re
import json
import logging
from tqdm import tqdm
from typing import List

from .tagging_basics import get_all_singular_plural_forms
from .captioning import CaptionGenerator
from ..character import Character
from ..basics import get_images_recursively, get_corr_meta_names


def _split_to_words(text: str):
    """
    Split a string into words separated by whitespaces and underscores.

    Args:
        text (str): The input text to split.

    Returns:
        List[str]: A list of lowercase words from the input text.
    """
    words = [subtext.split("_") for subtext in text.split()]
    # return [word.lower() for word in re.split(r'[\W_]+', text) if word]
    return [word.lower() for sublist in words for word in sublist if word]


def _match_suffix(tag: str, suffix: str):
    """
    Check if a tag matches a given suffix.

    Args:
        tag (str): The tag to check.
        suffix (str): The suffix to match.

    Returns:
        bool: True if the tag matches the suffix, False otherwise.
    """
    tag_words = _split_to_words(tag)
    suffix_words = _split_to_words(suffix)

    return tag_words[-len(suffix_words) :] == suffix_words


def _match_prefix(tag: str, prefix: str):
    """
    Check if a tag matches a given prefix.

    Args:
        tag (str): The tag to check.
        prefix (str): The prefix to match.

    Returns:
        bool: True if the tag matches the prefix, False otherwise.
    """
    tag_words = _split_to_words(tag)
    prefix_words = _split_to_words(prefix)
    return tag_words[: len(prefix_words)] == prefix_words


class CharacterTagProcessor(object):
    def __init__(
        self,
        tag_list_path: str,
        drop_difficulty: int = 2,
        emb_min_difficulty: int = 1,
        emb_max_difficutly: int = 2,
        drop_all: bool = False,
        emb_init_all: bool = False,
    ):
        """
        Generates default character tag lists based on the specified
        difficulties.

        This function returns lists of tags that should be whitelisted,
        dropped, or used for embedding initialization based on the provided
        difficulty levels.

        Args:
            tag_list_path (str):
                The path to the JSON file containing the tag lists.
            drop_difficulty (int):
                The difficulty level up to which tags should be dropped. Tags with
                difficulty less than this value will be added to the drop lists.
                Defaults to 2.
            emb_min_difficulty (int):
                The difficulty level from which tags should be used for embedding
                initialization.
                Defaults to 1.
            emb_max_difficutly (int):
                The difficulty level up to which tags should be used for embedding
                initialization.
                Defaults to 2.
            drop_all (bool):
                Whether to drop all core tags.
            emb_init_all (bool):
                Whether to use all core tags for embedding initialization.

        Initialize from a JSON file to generate:
            - whitelist: List of tags that are always whitelisted.
            - drop_prefixes: List of tag prefixes to be dropped.
            - drop_suffixes: List of tag suffixes to be dropped.
            - emb_init_prefixes: List of tag prefixes for embedding
            initialization.
            - emb_init_suffixes: List of tag suffixes for embedding
            initialization.
        """
        with open(tag_list_path, "r") as f:
            tag_lists = json.load(f)
            self._char_whitelist = tag_lists["whitelist"]
            self._char_prefixes = tag_lists["prefixes"]
            self._char_suffixes = tag_lists["suffixes"]

        self.whitelist = get_all_singular_plural_forms(self._char_whitelist)
        self.drop_prefixes = []
        self.drop_suffixes = []
        self.drop_all = drop_all
        self.emb_init_prefixes = []
        self.emb_init_suffixes = []
        self.emb_init_all = emb_init_all

        for difficulty in range(0, drop_difficulty):
            if difficulty <= len(self._char_prefixes):
                self.drop_prefixes.extend(self._char_prefixes[difficulty])
            if difficulty <= len(self._char_suffixes):
                self.drop_suffixes.extend(self._char_suffixes[difficulty])

        for difficulty in range(emb_min_difficulty, emb_max_difficutly):
            if difficulty <= len(self._char_prefixes):
                self.emb_init_prefixes.extend(self._char_prefixes[difficulty])
            if difficulty <= len(self._char_suffixes):
                self.emb_init_suffixes.extend(self._char_suffixes[difficulty])

        self.drop_suffixes = get_all_singular_plural_forms(self.drop_suffixes)
        self.emb_init_suffixes = get_all_singular_plural_forms(self.emb_init_suffixes)

    def is_character_tag(self, tag, mode="drop"):
        """
        Check if a tag is a basic character tag
        by matching with predefined whitelisted and blacklisted patterns.

        Args:
            tag (str): The tag to check.
        Returns:
            bool: True if the tag is a basic character tag, False otherwise.
        """
        assert mode in ["drop", "emb_init"]
        if mode == "drop" and self.drop_all:
            return True
        if mode == "emb_init" and self.emb_init_all:
            return True
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

        This method filters out character tags from a provided list or dictionary
        (mapping) of tags.

        Args:
            tags (Union[List[str], Mapping[str, float]]):
                The tags to be filtered. Can be either a list of strings (tag names) or
                a dictionary with tags as keys and their corresponding values
                (e.g., probabilities or scores).

        Returns:
            Union[List[str], Mapping[str, float]]:
                The filtered tags, in the same format as the input
                (either a list or a dictionary).

        Raises:
            TypeError: If the input 'tags' is neither a list nor a dictionary.
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


class CoreTagProcessor(object):
    """
    A class designed to identify and manage core tags for characters.
    The core tags are determined by their appearance frequency in a dataset.

    Attributes:
        frequency_threshold (float):
            The minimum frequency for a tag to be considered core tag.
        core_tags (dict):
            A dictionary where keys are characters and values are (lists of) core tags.
        emb_init_tags (dict):
            A dictionary containing tags to initialize embeddings.
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
        "wall",
        "tree",
        "cloud",
        "day",
        "night",
        "indoors",
        "outdoors",
        "close-up",
        "window",
        "curtains",
    ]

    def __init__(self, folder_path=None, core_tag_path=None, frequency_threshold=0.4):
        """
        For each character in the given folder, find the tags whose appearance
        frequency is higher than a certain threshold.
        A character is considered only if there contains solo images of the character.

        Args:
            folder_path (str, optional):
                The path to the folder containing image metadata.
            core_tag_path (str, optional):
                The path to a pre-existing file of core tags.
            frequency_threshold (float):
                The minimum frequency for a tag to be considered core tag.
        """
        assert folder_path is not None or core_tag_path is not None

        self.frequency_threshold = frequency_threshold
        if core_tag_path:
            with open(core_tag_path, "r") as core_tag_file:
                self.core_tags = json.load(core_tag_file)
        else:
            img_paths = get_images_recursively(folder_path)
            self.character_tag_dict = dict()
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
                    self._update_character_tag_dict(characters[0], tags)
            self.core_tags = self._get_frequent_tags()

    # TODO: Improved the blacklist mechanism here (maybe use whitelist instead)
    # for example mole under mouth should be retained
    def _contains_blacklisted_word_core(self, tag: str):
        """
        Check if a tag contains any blacklisted words.

        Args:
            tag (str): The tag to check.

        Returns:
            bool: True if the tag contains a blacklisted word, False otherwise.
        """
        words = [word for word in re.split(r"[\W_]+", tag.lower()) if word]
        return any((word in self._BLACKLISTED_WORDS_CORE) for word in words)

    def _update_character_tag_dict(
        self,
        character: str,
        tags: List[str],
    ):
        """
        Update the tag dictionary for a given character with new tags.

        Args:
            character (str): The character string representation.
            tags (List[str]): A list of tags associated with the character.
        """
        if character not in self.character_tag_dict:
            self.character_tag_dict[character] = [0, dict()]
        for tag in tags:
            if tag not in self.character_tag_dict[character][1]:
                self.character_tag_dict[character][1][tag] = 1
            else:
                self.character_tag_dict[character][1][tag] += 1
        self.character_tag_dict[character][0] += 1

    def _get_frequent_tags(self):
        """
        Extract tags for each character that appear more frequently
        than the given threshold.

        Returns:
            dict: A dictionary where keys are characters and values are lists
            of frequent tags.
        """
        frequent_tags = {}
        for character, (total, tags) in self.character_tag_dict.items():
            sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
            frequent_tags[character] = [
                tag
                for tag, count in sorted_tags
                if count >= self.frequency_threshold * total
                and not self._contains_blacklisted_word_core(tag)
            ]
        return frequent_tags

    def get_core_tags(self):
        """
        Get the dictionary of core tags.

        Returns:
            dict: The core tags dictionary.
        """
        return self.core_tags

    def save_core_tags(
        self,
        json_output: str,
        wildcard_ouput: str,
        caption_generator: CaptionGenerator,
    ):
        """
        Save the core tags to a JSON file and a wildcard output file.

        Args:
            json_output (str): The file path for saving the JSON output.
            wildcard_output (str): The file path for saving the wildcard output.
            caption_generator (CaptionGenerator): The caption generator.
        """
        with open(json_output, "w") as f:
            json.dump(self.core_tags, f, indent=4)
        with open(wildcard_ouput, "w") as f:
            for character, tags in self.core_tags.items():
                meta = {
                    "characters": [character],
                }
                caption = caption_generator.generate_caption(meta)
                f.write(caption + "\n")
                # The case where we split into kept, dropped, and emb_init
                if isinstance(tags, dict):
                    tags = tags["kept"]
                if len(tags) > 0:
                    meta["tags"] = tags
                    caption = caption_generator.generate_caption(meta)
                    f.write(caption + "\n")

    def drop_character_core_tags(self, characters, tags):
        """
        Drop tags from a list based on the character core tags.

        Args:
            characters (List[str]): List of characters.
            tags (List[str]): List of tags to be filtered.

        Returns:
            List[str]: Filtered list of tags.
        """
        filtered_tags = tags.copy()
        to_drop = []
        for character in characters:
            if character not in self.core_tags:
                logging.warning(
                    f"Character '{character}' not found in core tag dictionary."
                )
                continue

            # Get the tags to be dropped for this character
            to_drop.extend(
                self.core_tags[character].get("dropped", [])
                + self.core_tags[character].get("emb_init", [])
            )

        # Remove the tags from the filtered_tags list
        filtered_tags = [tag for tag in filtered_tags if tag not in to_drop]

        return filtered_tags

    def categorize_core_tags(self, char_tag_proc: CharacterTagProcessor):
        """
        Categorize and update core tags based on shared embeddings and categorization
        by a provided CharacterTagProcessor.

        Args:
            char_tag_proc (CharacterTagProcessor):
                An instance of CharacterTagProcessor used for categorizing tags.

        Returns:
            Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
                - The first dictionary contains categorized tags at the
                  character form level.
                - The second dictionary contains 'emb_init' tags at the embedding level.
        """
        # Categorize tags for each character form initially
        categorized_character_tags = {
            character: char_tag_proc.categorize_tags(tags)
            for character, tags in self.core_tags.items()
        }

        # Group characters by embedding
        embedding_dict = {}
        for character, categorized_tags in categorized_character_tags.items():
            embedding = Character.from_string(character).embedding_name
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

        self.core_tags = character_form_dict
        self.emb_init_tags = embedding_level_dict

        return character_form_dict, embedding_level_dict
