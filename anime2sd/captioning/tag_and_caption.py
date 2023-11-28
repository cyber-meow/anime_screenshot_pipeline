import os
import json
import logging
from typing import List

from waifuc.action import TaggingAction

from .tagging_character import CoreTagProcessor
from .waifuc_actions import (
    TagPruningAction,
    TagSortingAction,
    TagRemovingUnderscoreAction,
    CoreCharacterTagPruningAction,
    CaptioningAction,
)
from .captioning import CaptionGenerator

from ..emb_utils import update_emb_init_info
from ..waifuc_customize import LocalSource, SaveExporter


class TaggingManager(object):
    """
    Configuration manager for image tagging and tag pruning processes.

    This class encapsulates various settings and parameters used in the process of
    tagging images and pruning tags. It also includes methods for generating specific
    actions based on the provided configurations.

    Attributes:
        tagging_method (str):
            The method used for tagging images.
            Options are 'deepdanbooru', 'wd14_vit', 'wd14_convnext',
            'wd14_convnextv2', 'wd14_swinv2', 'mldanbooru'
        tag_threshold (float):
            The threshold value for tag selection.
        overwrite_tags (bool):
            Flag to indicate whether existing tags should be overwritten.
        pruned_mode (str):
            The mode of tag pruning to apply.
            Options are 'character', 'character_core', 'minimal', 'none'.
        character_tag_processor (CharacterTagProcessor):
            Processor for character-specific tag handling.
        process_from_original_tags (bool):
            Flag to indicate whether to process from original tags.
        sort_mode (str):
            The mode for sorting tags.
            Options are 'score', 'shuffle', 'original'.
        max_tag_number (int):
            The maximum number of tags to retain.
        blacklisted_tags (set):
            A set of tags that are to be excluded.
        overlap_tags_dict (dict):
            A dictionary mapping tags to their overlapped tags.
    """

    def __init__(
        self,
        tagging_method,
        tag_threshold,
        overwrite_tags,
        pruned_mode,
        blacklist_tags_file,
        overlap_tags_file,
        character_tag_processor,
        process_from_original_tags,
        sort_mode,
        max_tag_number,
    ):
        self.tagging_method = tagging_method
        self.tag_threshold = tag_threshold
        self.overwrite_tags = overwrite_tags
        self.pruned_mode = pruned_mode
        self.character_tag_processor = character_tag_processor
        self.process_from_original_tags = process_from_original_tags
        self.sort_mode = sort_mode
        self.max_tag_number = max_tag_number

        with open(blacklist_tags_file, "r") as f:
            self.blacklisted_tags = {line.strip() for line in f}
        self.overlap_tags_dict = self._parse_overlap_tags(overlap_tags_file)

    @staticmethod
    def _parse_overlap_tags(json_file):
        """
        Parses a JSON file to extract overlapping tag information.

        Args:
            json_file (str): Path to the JSON file containing overlap tag data.

        Returns:
            dict: A dictionary with queries as keys and their overlapped tags as values.
        """
        with open(json_file, "r") as file:
            data = json.load(file)

        overlap_tags_dict = {
            entry["query"]: entry["has_overlap"]
            for entry in data
            if "has_overlap" in entry and entry["has_overlap"]
        }

        return overlap_tags_dict

    def get_tagging_action(self):
        """
        Creates and returns a TaggingAction instance based on the current configuration.

        Returns:
            TaggingAction:
                An instance for tagging as per the current settings.
        """
        return TaggingAction(
            force=self.overwrite_tags,
            method=self.tagging_method,
            general_threshold=self.tag_threshold,
            character_threshold=1.01,
        )

    def get_basic_tag_pruning_action(self):
        """
        Creates and returns a TagPruningAction instance based on the
        current configuration.

        Returns:
            TagPruningAction:
                An instance for pruning tags as per the current settings.
        """
        if self.process_from_original_tags or self.overwrite_tags:
            tags_attribute = "tags"
        else:
            tags_attribute = "processed_tags"
        if self.pruned_mode == "character_core":
            pruned_mode = "minimal"
        else:
            pruned_mode = self.pruned_mode
        return TagPruningAction(
            self.blacklisted_tags,
            self.overlap_tags_dict,
            pruned_mode=pruned_mode,
            tags_attribute=tags_attribute,
            character_tag_processor=self.character_tag_processor,
        )

    def get_tag_sorting_action(self):
        """
        Creates and returns a TagSortingAction instance based on the
        current configuration.

        Returns:
            TagSortingAction:
                An instance for sorting tags as per the current settings.
        """
        return TagSortingAction(self.sort_mode, max_tag_number=self.max_tag_number)


def tag_and_caption_from_directory(
    dir: str,
    tagging_manager: TaggingManager,
    caption_generator: CaptionGenerator,
    use_existing_core_tag_file: bool,
    core_frequency_threshold: float,
    image_type: str,
    overwrite_emb_init_info: bool,
    load_aux: List[str],
    save_aux: List[str],
    overwrite_path: bool,
):
    """
    Processes images in a directory for tagging and captioning.

    This function takes a directory of images and applies several processing
    steps including tagging, tag pruning, captioning, and saving the processed
    data. It handles both general tags and character-specific tags, and allows
    for various customizations such as sorting mode, tag threshold, and using
    probabilities in captioning.

    Args:
        dir (str):
            Path to the directory containing images.
        tagging_manager (TaggingManager):
            The tagging manager for managing tag operations.
        caption_generator (CaptionGenerator):
            The caption generator for generating captions.
        use_existing_core_tag_file (bool):
            Flag to use existing core tags file.
        core_frequency_threshold (float):
            Frequency threshold for core tags.
        image_type (str):
            Type of images being processed (for embedding initialization).
        overwrite_emb_init_info (bool):
            Flag to overwrite embedding initialization info.
        load_aux (list):
            List of auxiliary attributes to load.
        save_aux (list):
            List of auxiliary attributes to save.
        overwrite_path (bool):
            Whether to overwrite path in metadata or not.

    Returns:
        None
    """
    core_tag_path = os.path.join(dir, "core_tags.json")
    wildcard_path = os.path.join(dir, "wildcard.txt")
    emb_init_filepath = os.path.join(dir, "emb_init.json")

    source = LocalSource(dir, load_aux=load_aux, overwrite_path=overwrite_path)
    source = source.attach(
        tagging_manager.get_tagging_action(),
        tagging_manager.get_basic_tag_pruning_action(),
        TagRemovingUnderscoreAction(),
    )

    if tagging_manager.pruned_mode == "character_core":
        source.export(
            SaveExporter(dir, no_meta=False, save_caption=False, in_place=True)
        )
        if use_existing_core_tag_file:
            core_tag_processor = CoreTagProcessor(
                core_tag_path=core_tag_path,
            )
        else:
            assert tagging_manager.character_tag_processor is not None
            core_tag_processor = CoreTagProcessor(
                folder_path=dir, frequency_threshold=core_frequency_threshold
            )
            emb_init_dict = core_tag_processor.categorize_core_tags(
                tagging_manager.character_tag_processor
            )[1]
            core_tag_processor.save_core_tags(
                core_tag_path,
                wildcard_path,
                caption_generator,
            )
            update_emb_init_info(
                emb_init_filepath,
                emb_init_dict.keys(),
                image_type,
                emb_init_dict=emb_init_dict,
                overwrite=overwrite_emb_init_info,
            )
        source = source.attach(
            CoreCharacterTagPruningAction(
                core_tag_processor, tags_attribute="processed_tags"
            )
        )
        characters = list(core_tag_processor.get_core_tags().keys())
    else:
        characters = None

    source = source.attach(
        tagging_manager.get_tag_sorting_action(),
        CaptioningAction(caption_generator, characters),
    )
    source.export(
        SaveExporter(
            dir,
            no_meta=False,
            save_caption=True,
            save_aux=save_aux,
            in_place=True,
        )
    )

    if tagging_manager.pruned_mode != "character_core":
        core_tag_processor = CoreTagProcessor(
            dir, frequency_threshold=core_frequency_threshold
        )
        core_tag_processor.save_core_tags(
            core_tag_path,
            wildcard_path,
            caption_generator,
        )
