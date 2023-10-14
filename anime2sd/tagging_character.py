import re
from hbutils.string import singular_form, plural_form


_CHAR_WHITELIST = [
    'drill', 'pubic hair', 'closed eyes', 'half-closed eyes', 'empty `eyes'
]


_CHAR_SUFFIXES_EASY = [
    'eyes', 'skin', 'hair', 'bun', 'bangs', 'cut', 'sidelocks',
    'twintails', 'braid', 'braids', 'afro', 'ahoge', 'drill',
    'drills', 'bald', 'dreadlocks', 'side up', 'ponytail', 'updo',
    'beard', 'mustache', 'hair intake',
]


_CHAR_SUFFIXES_HARD = [
    'ear', 'horn', 'halo'
]


_CHAR_PREFIXES = [
    'hair over', 'hair between'
]


def _split_to_words(text: str):
    """
    Split a string into words and return them in lowercase.

    :param text: The input text to split.
    :type text: str
    :return: List of lowercase words.
    :rtype: List[str]
    """
    return [word.lower() for word in re.split(r'[\W_]+', text) if word]


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
    all_suffixes = [suffix_words]
    all_suffixes.append([*suffix_words[:-1], singular_form(suffix_words[0])])
    all_suffixes.append([*suffix_words[:-1], plural_form(suffix_words[0])])

    for suf in all_suffixes:
        if tag_words[-len(suf):] == suf:
            return True

    return False


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


def _match_same(tag: str, expected: str):
    """
    Check if a tag matches another tag, considering singular and plural forms.

    :param tag: The tag to check.
    :type tag: str
    :param expected: The expected tag.
    :type expected: str
    :return: True if the tag matches the expected tag, False otherwise.
    :rtype: bool
    """
    a = _split_to_words(tag)
    as_ = [a, [*a[:-1], singular_form(a[-1])], [*a[:-1], plural_form(a[-1])]]
    as_ = set([tuple(item) for item in as_])

    b = _split_to_words(expected)
    bs_ = [b, [*b[:-1], singular_form(b[-1])], [*b[:-1], plural_form(b[-1])]]
    bs_ = set([tuple(item) for item in bs_])

    return bool(as_ & bs_)


def is_basic_character_tag(tag, whitelist, prefixes, suffixes):
    """
    Check if a tag is a basic character tag
    by matching with predefined whitelisted and blacklisted patterns.

    :param tag: The tag to check.
    :type tag: str
    :return: True if the tag is a basic character tag, False otherwise.
    :rtype: bool
    """
    if any(_match_same(tag, wl_tag) for wl_tag in whitelist):
        return False
    else:
        return (any(_match_suffix(tag, suffix) for suffix in suffixes)
                or any(_match_prefix(tag, prefix) for prefix in prefixes))


def drop_basic_character_tags(tags, drop_hard):
    """
    Drop basic character tags from the given list or mapping of tags.

    :param tags: List or mapping of tags to be filtered.
    :type tags: Union[List[str], Mapping[str, float]]
    :return: Filtered list or mapping of tags without the basic character tags.
    :rtype: Union[List[str], Mapping[str, float]]
    :raises TypeError: If the input tags are neither a list nor a dictionary.
    """
    whitelist = _CHAR_WHITELIST
    prefixes = _CHAR_PREFIXES
    suffixes = _CHAR_SUFFIXES_EASY
    if drop_hard:
        suffixes.extend(_CHAR_SUFFIXES_HARD)
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
