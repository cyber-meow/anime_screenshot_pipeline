import random
from typing import List, Optional

from ..character import Character


class CaptionGenerator(object):
    """
    A class for generating captions based on image metadata.

    Manages the creation of captions by incorporating different types of information
    based on their respective probabilities and formatting.
    """

    def __init__(
        self,
        character_sep: str,
        character_inner_sep: str,
        character_outer_sep: str,
        caption_inner_sep: str,
        caption_outer_sep: str,
        use_npeople_prob: float,
        use_character_prob: float,
        use_copyright_prob: float,
        use_image_type_prob: float,
        use_artist_prob: float,
        use_rating_prob: float,
        use_tags_prob: float,
    ):
        """
        Initializes the CaptionGenerator with individual separators and probabilities
        for each type of information.

        Args:
            character_sep (str): Separator for characters.
            character_inner_sep (str): Inner separator for character information.
            character_outer_sep (str): Outer separator for character information.
            caption_inner_sep (str): Inner separator for captions.
            caption_outer_sep (str): Outer separator for captions.
            use_npeople_prob (float): Probability of using number of people information.
            use_character_prob (float): Probability of using character information.
            use_copyright_prob (float): Probability of using copyright information.
            use_image_type_prob (float): Probability of using image type information.
            use_artist_prob (float): Probability of using artist information.
            use_rating_prob (float): Probability of using rating information.
            use_tags_prob (float): Probability of using tags information.
        """
        self.character_sep = character_sep
        self.character_inner_sep = character_inner_sep
        self.character_outer_sep = character_outer_sep
        self.caption_inner_sep = caption_inner_sep
        self.caption_outer_sep = caption_outer_sep
        self.use_npeople_prob = use_npeople_prob
        self.use_characters_prob = use_character_prob
        self.use_copyright_prob = use_copyright_prob
        self.use_type_prob = use_image_type_prob
        self.use_artist_prob = use_artist_prob
        self.use_rating_prob = use_rating_prob
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
        if attribute == "processed_tags":
            prob_attr = "use_tags_prob"
        else:
            prob_attr = f"use_{attribute}_prob"
        if not hasattr(self, prob_attr):
            raise AttributeError(
                f"Probability attribute '{prob_attr}' not found in CaptionGenerator."
            )
        prob = getattr(self, prob_attr)

        if random.random() >= prob or attribute not in info_dict:
            return caption
        to_add = info_dict[attribute]
        if not to_add:
            return caption
        if characters is None:
            to_add_text = to_text_method(to_add)
        else:
            to_add_text = to_text_method(to_add, characters)
        if to_add_text is not None:
            if caption != "":
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
        if to_add == "explicit":
            return "explicit"
        else:
            return None

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
        }

        for attribute, to_text_method in captioning_methods.items():
            characters_tmp = None if attribute != "characters" else characters
            caption = self.caption_add_content(
                caption, info_dict, attribute, to_text_method, characters_tmp
            )

        if "processed_tags" in info_dict:
            caption = self.caption_add_content(
                caption, info_dict, "processed_tags", self.to_text_tags
            )
        elif "tags" in info_dict:
            caption = self.caption_add_content(
                caption, info_dict, "tags", self.to_text_tags
            )
        return caption
