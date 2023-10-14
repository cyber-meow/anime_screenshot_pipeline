def _split_to_words(text: str):
    """
    Split a string into words separated by whitespaces and underscores

    :param text: The input text to split.
    :type text: str
    :return: List of lowercase words.
    :rtype: List[str]
    """
    words = [subtext.split('_') for subtext in text.split()]
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

    return tag_words[-len(suffix_words):] == suffix_words


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
    return tag_words[:len(prefix_words)] == prefix_words


def is_basic_character_tag(tag, whitelist, prefixes, suffixes):
    """
    Check if a tag is a basic character tag
    by matching with predefined whitelisted and blacklisted patterns.

    :param tag: The tag to check.
    :type tag: str
    :return: True if the tag is a basic character tag, False otherwise.
    :rtype: bool
    """
    if tag in whitelist:
        return False
    else:
        return (any(_match_suffix(tag, suffix) for suffix in suffixes)
                or any(_match_prefix(tag, prefix) for prefix in prefixes))


def drop_basic_character_tags(tags, whitelist, prefixes, suffixes):
    """
    Drop basic character tags from the given list or mapping of tags.

    :param tags: List or mapping of tags to be filtered.
    :type tags: Union[List[str], Mapping[str, float]]
    :return: Filtered list or mapping of tags without the basic character tags.
    :rtype: Union[List[str], Mapping[str, float]]
    :raises TypeError: If the input tags are neither a list nor a dictionary.
    """
    if isinstance(tags, dict):
        return {tag: value for tag, value in tags.items()
                if not is_basic_character_tag(
                    tag, whitelist, prefixes, suffixes)}
    elif isinstance(tags, list):
        return [tag for tag in tags
                if not is_basic_character_tag(
                    tag, whitelist, prefixes, suffixes)]
    else:
        raise TypeError(
            "Unsupported types of tags, dict or list expected, "
            f"but {tags!r} found.")
