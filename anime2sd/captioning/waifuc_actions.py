import logging
from typing import List, Optional

from waifuc.model import ImageItem
from waifuc.action.base import ProcessAction

from .tagging_basics import (
    drop_tags_from_dictionary,
    drop_blacklisted_tags,
    drop_overlap_tags,
    sort_tags,
)
from .tagging_character import CharacterTagProcessor, CoreTagProcessor
from .captioning import CaptionGenerator


class TagPruningAction(ProcessAction):
    def __init__(
        self,
        blacklisted_tags,
        overlap_tags_dict,
        prune_mode="character",
        tags_attribute="processed_tags",
        character_tag_processor: Optional[CharacterTagProcessor] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logging.getLogger() if logger is None else logger
        assert prune_mode in ["none", "minimal", "character"]
        self.blacklisted_tags = blacklisted_tags
        self.overlap_tags_dict = overlap_tags_dict
        self.prune_mode = prune_mode
        self.tags_attribute = tags_attribute
        if prune_mode == "character":
            assert character_tag_processor is not None
        self.character_tag_processor = character_tag_processor

    def process(self, item: ImageItem) -> ImageItem:
        if self.prune_mode == "none":
            return item
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            self.logger.warning(
                f"{self.tags_attribute} unfound "
                f"for {item.meta['current_path']}, skip"
            )
            return item
        # TODO: We should not deal with dictionary type within functions
        tags = drop_blacklisted_tags(tags, self.blacklisted_tags)
        tags = drop_overlap_tags(tags, self.overlap_tags_dict)
        if self.prune_mode == "character":
            assert self.character_tag_processor is not None
            # Only pruned character related tags for character images
            if "characters" in item.meta and item.meta["characters"]:
                (
                    kept_tags,
                    dropped_tags,
                ) = self.character_tag_processor.drop_character_tags(tags)
                kept_tags, dropped_tags = drop_tags_from_dictionary(
                    tags, kept_tags, dropped_tags
                )
                return ImageItem(
                    item.image,
                    {
                        **item.meta,
                        "processed_tags": kept_tags,
                        "dropped_character_tags": dropped_tags,
                    },
                )
        return ImageItem(item.image, {**item.meta, "processed_tags": tags})


class CoreCharacterTagPruningAction(ProcessAction):
    def __init__(
        self,
        core_tag_processor: CoreTagProcessor,
        tags_attribute: str = "processed_tags",
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logging.getLogger() if logger is None else logger
        self.tags_attribute = tags_attribute
        self.core_tag_processor = core_tag_processor

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            self.logger.warning(
                f"{self.tags_attribute} unfound "
                f"for {item.meta['current_path']}, skip"
            )
            return item
        # Only pruned character related tags for character images
        if "characters" in item.meta and item.meta["characters"]:
            kept_tags, dropped_tags = self.core_tag_processor.drop_character_core_tags(
                item.meta["characters"], tags
            )
            kept_tags, dropped_tags = drop_tags_from_dictionary(
                tags, kept_tags, dropped_tags
            )
            return ImageItem(
                item.image,
                {
                    **item.meta,
                    "processed_tags": kept_tags,
                    "dropped_character_tags": dropped_tags,
                },
            )
        return item


class TagSortingAction(ProcessAction):
    def __init__(
        self,
        sort_mode="score",
        max_tag_number: Optional[int] = None,
        append_dropped_character_tags: bool = False,
        tags_attribute: str = "processed_tags",
        logger: Optional[logging.Logger] = None,
    ):
        assert sort_mode in ["original", "shuffle", "score"]
        self.logger = logging.getLogger() if logger is None else logger
        self.sort_mode = sort_mode
        self.append_dropped_character_tags = append_dropped_character_tags
        self.max_tag_number = max_tag_number
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            self.logger.warning(
                f"{self.tags_attribute} unfound "
                f"for {item.meta['current_path']}, skip"
            )
            return item
        tags = sort_tags(tags, self.sort_mode)
        # A better way is to define the following as independent actions
        # Append dropped character tags if needed
        # Another solution is to delegate this to captioner but
        # this prevents from applying the maximum tag number constraint
        if self.append_dropped_character_tags and "dropped_character_tags" in item.meta:
            tags = tags + sort_tags(item.meta["dropped_character_tags"], self.sort_mode)
        # Trim to max_tag_number
        if self.max_tag_number and len(tags) > self.max_tag_number:
            tags = tags[: self.max_tag_number]
        return ImageItem(item.image, {**item.meta, "processed_tags": tags})


class TagRemovingUnderscoreAction(ProcessAction):
    def __init__(
        self, tags_attribute="processed_tags", logger: Optional[logging.Logger] = None
    ):
        self.logger = logging.getLogger() if logger is None else logger
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            self.logger.warning(
                f"{self.tags_attribute} unfound "
                f"for {item.meta['current_path']}, skip"
            )
            return item
        if isinstance(tags, list):
            tags = [self.remove_underscore(tag) for tag in tags]
        elif isinstance(tags, dict):
            tags = {self.remove_underscore(key): value for key, value in tags.items()}
        else:
            raise ValueError(f"Unsupported type of tags: {type(tags)}")
        result = ImageItem(item.image, {**item.meta, "processed_tags": tags})
        return result

    @staticmethod
    def remove_underscore(tag):
        if tag == "^_^":
            return tag
        return tag.replace("_", " ")


class CaptioningAction(ProcessAction):
    def __init__(
        self,
        caption_generator: CaptionGenerator,
        characters: Optional[List[str]] = None,
    ):
        self.caption_generator = caption_generator
        self.characters = characters

    def process(self, item: ImageItem) -> ImageItem:
        caption = self.caption_generator.generate_caption(item.meta, self.characters)
        return ImageItem(item.image, {**item.meta, "caption": caption})
