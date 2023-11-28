import logging
from typing import List, Optional

from waifuc.model import ImageItem
from waifuc.action.base import ProcessAction

from .tagging_basics import drop_blacklisted_tags, drop_overlap_tags, sort_tags
from .tagging_character import CharacterTagProcessor, CoreTagProcessor
from .captioning import CaptionGenerator


class TagPruningAction(ProcessAction):
    def __init__(
        self,
        blacklisted_tags,
        overlap_tags_dict,
        pruned_mode="character",
        tags_attribute="processed_tags",
        character_tag_processor: Optional[CharacterTagProcessor] = None,
    ):
        assert pruned_mode in ["none", "minimal", "character"]
        self.blacklisted_tags = blacklisted_tags
        self.overlap_tags_dict = overlap_tags_dict
        self.pruned_mode = pruned_mode
        self.tags_attribute = tags_attribute
        if pruned_mode == "character":
            assert character_tag_processor is not None
        self.character_tag_processor = character_tag_processor

    def process(self, item: ImageItem) -> ImageItem:
        if self.pruned_mode == "none":
            return item
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip",
            )
            return item
        tags = drop_blacklisted_tags(tags, self.blacklisted_tags)
        tags = drop_overlap_tags(tags, self.overlap_tags_dict)
        if self.pruned_mode == "character":
            assert self.character_tag_processor is not None
            # Only pruned character related tags for character images
            if "characters" in item.meta and item.meta["characters"]:
                tags = self.character_tag_processor.drop_character_tags(tags)
        return ImageItem(item.image, {**item.meta, "processed_tags": tags})


class CoreCharacterTagPruningAction(ProcessAction):
    def __init__(
        self,
        core_tag_processor: CoreTagProcessor,
        tags_attribute: str = "processed_tags",
    ):
        self.tags_attribute = tags_attribute
        self.core_tag_processor = core_tag_processor

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip",
            )
            return item
        # Only pruned character related tags for character images
        if "characters" in item.meta and item.meta["characters"]:
            tags = self.core_tag_processor.drop_character_core_tags(
                item.meta["characters"], tags
            )
        return ImageItem(item.image, {**item.meta, "processed_tags": tags})


class TagSortingAction(ProcessAction):
    def __init__(
        self, sort_mode="score", max_tag_number=None, tags_attribute="processed_tags"
    ):
        assert sort_mode in ["original", "shuffle", "score"]
        self.sort_mode = sort_mode
        self.max_tag_number = max_tag_number
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip",
            )
            return item
        tags = sort_tags(tags, self.sort_mode)
        if self.max_tag_number is not None and len(tags) > self.max_tag_number:
            tags = tags[: self.max_tag_number]
        return ImageItem(item.image, {**item.meta, "processed_tags": tags})


class TagRemovingUnderscoreAction(ProcessAction):
    def __init__(self, tags_attribute="processed_tags"):
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif "tags" in item.meta:
            tags = item.meta["tags"]
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip",
            )
            return item
        tags = [self.remove_underscore(tag) for tag in tags]
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
