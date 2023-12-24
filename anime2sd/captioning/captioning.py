import random
from typing import List, Optional

from ..character import Character


class CaptionGenerator(object):
    """
    A class for generating captions based on image metadata.

    Manages the creation of captions by incorporating different types of information
    based on their respective probabilities and formatting.
    """

    _ATTRIBUTE_MAPPING = {
        "npeople": "npeople",
        "n_people": "npeople",
        "character": "characters",
        "characters": "characters",
        "copyright": "copyright",
        "copyrights": "copyright",
        "type": "type",
        "image_type": "type",
        "artist": "artist",
        "artists": "artist",
        "rating": "rating",
        "crop_info": "crop",
        "crop": "crop",
        "tag": "tags",
        "tags": "tags",
    }

    def __init__(
        self,
        character_sep: str = ", ",
        character_inner_sep: str = " ",
        character_outer_sep: str = ", ",
        caption_inner_sep: str = ", ",
        caption_outer_sep: str = ", ",
        keep_tokens_sep: Optional[str] = None,
        keep_tokens_before: str = "tags",
        caption_ordering: Optional[List[str]] = None,
        use_npeople_prob: float = 0,
        use_character_prob: float = 1,
        use_copyright_prob: float = 1,
        use_image_type_prob: float = 1,
        use_artist_prob: float = 1,
        use_rating_prob: float = 1,
        use_crop_info_prob: float = 1,
        use_tags_prob: float = 1,
    ):
        """
        Initializes the CaptionGenerator with individual separators and probabilities
        for each type of information.

        The default caption ordering is
        ['npeople', 'character', 'copyright', 'image_type',
         'artist', 'rating', 'crop_info', tags'].

        Args:
            character_sep (str):
                Separator for characters.
            character_inner_sep (str):
                Inner separator for character information.
            character_outer_sep (str):
                Outer separator for character information.
            caption_inner_sep (str):
                Inner separator for captions.
            caption_outer_sep (str):
                Outer separator for captions.
            keep_tokens_sep (str):
                Separator for keep tokens for Kohya trainer.
                Defaults to None which uses `character_outer_sep`.
            keep_tokens_before (str):
                Where to put `keep_tokens_sep` before.
                Should be one of attributes listed in the default caption ordering.
                Defaults to 'tags'.
            caption_ordering (List[str]):
                A list of attributes indicating the order of captions.
                Defaults to None which uses the default caption ordering.
            use_npeople_prob (float):
                Probability of using number of people information.
            use_character_prob (float):
                Probability of using character information.
            use_copyright_prob (float):
                Probability of using copyright information.
            use_image_type_prob (float):
                Probability of using image type information.
            use_artist_prob (float):
                Probability of using artist information.
            use_rating_prob (float):
                Probability of using rating information.
            use_crop_info_prob (float):
                Probability of using crop information.
            use_tags_prob (float):
                Probability of using tags information.
        """
        self.character_sep = character_sep
        self.character_inner_sep = character_inner_sep
        self.character_outer_sep = character_outer_sep
        self.caption_inner_sep = caption_inner_sep
        self.caption_outer_sep = caption_outer_sep

        self.keep_tokens_sep = keep_tokens_sep or character_outer_sep
        self.keep_tokens_before = self._ATTRIBUTE_MAPPING.get(
            keep_tokens_before, keep_tokens_before
        )

        default_ordering = [
            "npeople",
            "character",
            "copyright",
            "image_type",
            "artist",
            "rating",
            "crop_info",
            "tags",
        ]
        caption_ordering = caption_ordering or default_ordering
        self.caption_ordering = []
        for attribute in caption_ordering:
            if attribute not in self._ATTRIBUTE_MAPPING:
                raise ValueError(
                    f"Attribute {attribute} is not a valid attribute for captioning."
                )
            attribute = self._ATTRIBUTE_MAPPING[attribute]
            self.caption_ordering.append(attribute)

        self.use_npeople_prob = use_npeople_prob
        self.use_characters_prob = use_character_prob
        self.use_copyright_prob = use_copyright_prob
        self.use_type_prob = use_image_type_prob
        self.use_artist_prob = use_artist_prob
        self.use_rating_prob = use_rating_prob
        self.use_crop_prob = use_crop_info_prob
        self.use_tags_prob = use_tags_prob

    def caption_add_content(
        self, caption, info_dict, attribute, to_text_method, characters=None
    ):
        """
        Adds content to the caption based on the specified attribute.

        Args:
            caption (str):
                The existing caption.
            info_dict (dict):
                A dictionary containing various image attributes.
            attribute (str):
                The attribute to consider for adding to the caption.
            to_text_method (callable):
                A method to convert the attribute to text.
            characters (list, optional):
                A list of characters for filtering, if applicable.

        Returns:
            str: The updated caption.
        """
        prob_attr = f"use_{attribute}_prob"
        if not hasattr(self, prob_attr):
            raise AttributeError(
                f"Probability attribute '{prob_attr}' not found in CaptionGenerator."
            )
        prob = getattr(self, prob_attr)

        if random.random() >= prob or attribute not in info_dict:
            return caption

        if attribute == "tags" and "processed_tags" in info_dict:
            to_add = info_dict["processed_tags"]
        else:
            to_add = info_dict[attribute]
        if not to_add:
            return caption
        if characters is None:
            to_add_text = to_text_method(to_add)
        else:
            to_add_text = to_text_method(to_add, characters)
        if to_add_text is not None:
            if caption != "":
                if self.keep_tokens_before == attribute:
                    caption += self.keep_tokens_sep
                else:
                    caption += self.caption_outer_sep
            caption += to_add_text
        return caption

    def to_text_npeople(self, count):
        suffix = "person" if count == 1 else "people"
        return f"{count}{suffix}"

    def to_text_characters(self, to_add, characters=None):
        if not isinstance(to_add, list):
            to_add = [to_add]
        # Ideally this should be filtered in earlier stages
        # Still left here just in case
        if characters is None:
            to_add = list(filter(lambda item: item != "unknown", to_add))
        else:
            to_add = list(filter(lambda item: item in characters, to_add))
        to_add = [
            Character.from_string(character).to_string(
                inner_sep=self.character_inner_sep,
                outer_sep=self.character_outer_sep,
                caption_style=True,
            )
            for character in to_add
        ]
        return self.character_sep.join(to_add)

    def to_text_copyright(self, to_add):
        if not isinstance(to_add, list):
            to_add = [to_add]
        to_add = list(filter(lambda item: item != "unknown", to_add))
        return "from " + self.caption_inner_sep.join(to_add)

    def to_text_type(self, image_type):
        return image_type

    def to_text_artist(self, to_add):
        if not isinstance(to_add, list):
            to_add = [to_add]
        to_add = list(filter(lambda item: item != "anonymous", to_add))
        to_add = list(filter(lambda item: item != "unknown", to_add))
        return "by " + self.caption_inner_sep.join(to_add)

    def to_text_rating(self, to_add):
        if to_add in ["explicit", "e"]:
            return "explicit"
        else:
            return None

    def to_text_crop(self, to_add):
        if isinstance(to_add, dict) and "type" in to_add:
            return to_add["type"] + " cropped"
        return "cropped"

    def to_text_tags(self, to_add):
        # Case of {tag: score}
        if isinstance(to_add, dict):
            to_add = to_add.keys()
        to_add = list(filter(lambda item: item != "unknown", to_add))
        return self.caption_inner_sep.join(to_add)

    def generate_caption(self, info_dict: dict, characters: Optional[List[str]] = None):
        """
        Converts a dictionary of image metadata into a caption.

        Args:
            info_dict (dict): A dictionary containing image metadata.
            characters (list): A list of characters for filtering, if applicable.

        Returns:
            str: The generated caption.
        """
        caption = ""
        captioning_methods = {
            "npeople": self.to_text_npeople,
            "characters": self.to_text_characters,
            "copyright": self.to_text_copyright,
            "type": self.to_text_type,
            "artist": self.to_text_artist,
            "rating": self.to_text_rating,
            "crop": self.to_text_crop,
            "tags": self.to_text_tags,
        }

        for attribute in self.caption_ordering:
            to_text_method = captioning_methods[attribute]
            characters_tmp = None if attribute != "characters" else characters
            caption = self.caption_add_content(
                caption, info_dict, attribute, to_text_method, characters_tmp
            )
        return caption
