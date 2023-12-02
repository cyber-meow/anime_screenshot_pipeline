class Character(object):
    """
    Represents a character with various attributes including name, appearance,
    outfit, accessories, and objects.

    Attributes:
        character_name (str):
            The name of the character.
        appearance (str):
            The appearance or style of the character.
        outfit (list):
            List of outfits associated with the character.
        accessories (list):
            List of accessories associated with the character.
        objects (list):
            List of objects associated with the character.
        extra_features (list):
            List of extra attributes or features associated with the character.
    """

    def __init__(
        self,
        character_name,
        appearance=None,
        outfit=None,
        accessories=None,
        objects=None,
        extra_features=None,
        extra_embeddings=None,
    ):
        self.character_name = character_name
        self.appearance = appearance
        self.outfit = outfit if outfit is not None else []
        self.accessories = accessories if accessories is not None else []
        self.objects = objects if objects is not None else []
        self.extra_features = extra_features if extra_features is not None else []

        # Include appearance if it starts with "_"
        self.embedding_name = (
            f"{self.character_name}{self.appearance}"
            if self.appearance and self.appearance.startswith("_")
            else self.character_name
        )
        self.embedding_name = "_".join(self.embedding_name.split())
        self.extra_embeddings = extra_embeddings if extra_embeddings is not None else []

    @classmethod
    def _parse_attribute(cls, attribute_str, inner_sep, extra_embeddings):
        """
        Parses an attribute string into a list of items, handling special cases.

        Args:
            attribute_str (str): The attribute string to parse.
            inner_sep (str): The separator between the items of a list.
            extra_embeddings (list): List to append extra embeddings to.

        Returns:
            list or None: A list of parsed items or None if the string is empty.
        """
        if not attribute_str or attribute_str.lower() == "none":
            return None

        items = attribute_str.split(inner_sep)
        parsed_items = []
        for item in items:
            if item.startswith("_"):
                extra_embeddings.append(item[1:])
            parsed_items.append(item)
        return parsed_items

    @classmethod
    def from_string(cls, text, inner_sep="+", outer_sep="|"):
        """
        Initialize a Character object from a folder path.
        Names starting with "_" are treated as embeddings.

        Args:
            folder_path (str): Path to the folder containing character data.
            inner_sep (str): The separator between the items of a list.

        Returns:
            Character: An instance of the Character class.
        """
        parts = text.split(outer_sep)
        extra_embeddings = []
        character_name = parts[0]
        if len(parts) > 1 and parts[1].lower() != "none":
            appearance = parts[1]
        else:
            appearance = None

        outfits = cls._parse_attribute(
            parts[2] if len(parts) > 2 else "", inner_sep, extra_embeddings
        )
        accessories = cls._parse_attribute(
            parts[3] if len(parts) > 3 else "", inner_sep, extra_embeddings
        )
        objects = cls._parse_attribute(
            parts[4] if len(parts) > 4 else "", inner_sep, extra_embeddings
        )
        extra_features = cls._parse_attribute(
            parts[5] if len(parts) > 5 else "", inner_sep, extra_embeddings
        )

        return cls(
            character_name,
            appearance,
            outfits,
            accessories,
            objects,
            extra_features,
            extra_embeddings=extra_embeddings,
        )

    def to_string(self, inner_sep="+", outer_sep="|", caption_style=False):
        """Gets the text representation of the character

        Args:
            inner_sep (str): The inner separator between the attributes.
            outer_sep (str): The outer separator between the attributes.
            path_style (bool): Whether to format the output in caption style.

        Returns:
            str: The text representation of the character.
        """
        elements = []

        # Handle character name and appearance
        elements.append(self.character_name)
        if self.appearance:
            if caption_style:
                if self.appearance.startswith("_"):
                    elements[-1] = self.embedding_name
                else:
                    elements[-1] += inner_sep + self.appearance
            else:
                elements.append(self.appearance)
        elif not caption_style:
            elements.append("none")

        # Add attributes or placeholders
        attributes = [self.outfit, self.accessories, self.objects, self.extra_features]
        for attr in attributes:
            if attr:
                elements.append(inner_sep.join(attr))
            elif not caption_style:
                elements.append("none")

        # Remove trailing "None" placeholders if path_style is True
        if not caption_style:
            while elements and elements[-1] == "none":
                elements.pop()

        return outer_sep.join(elements)

    def __hash__(self):
        return hash(
            (
                self.character_name,
                self.appearance,
                tuple(self.outfit),
                tuple(self.accessories),
                tuple(self.objects),
                tuple(self.extra_features),
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Character):
            return NotImplemented
        return (
            self.character_name,
            self.appearance,
            self.outfit,
            self.accessories,
            self.objects,
            self.extra_features,
        ) == (
            other.character_name,
            other.appearance,
            other.outfit,
            other.accessories,
            other.objects,
            other.extra_features,
        )
