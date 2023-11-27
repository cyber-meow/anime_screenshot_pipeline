import random

from anime2sd.character import Character


def caption_add_content(
    caption,
    info_dict,
    attribute,
    prob,
    to_text,
    separators,
    characters=None,
):
    if random.random() >= prob or attribute not in info_dict:
        return caption
    to_add = info_dict[attribute]
    if not to_add:
        return caption
    to_add_text = to_text(to_add, separators, characters)
    if to_add_text is not None:
        if caption != "":
            caption += separators["caption_outer"]
        caption += to_add_text
    return caption


def to_text_npeople(count, separators, characters):
    suffix = "person" if count == 1 else "people"
    return f"{count}{suffix}"


def to_text_characters(to_add, separators, characters):
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
            inner_sep=separators["character_inner"],
            outer_sep=separators["character_outer"],
            caption_style=True,
        )
        for character in to_add
    ]
    return separators["character"].join(to_add)


def to_text_copyright(to_add, separators, characters):
    if not isinstance(to_add, list):
        to_add = [to_add]
    to_add = list(filter(lambda item: item != "unknown", to_add))
    return "from " + separators["caption_inner"].join(to_add)


def to_text_type(image_type, separators, characters):
    return image_type


def to_text_artist(to_add, separators, characters):
    if not isinstance(to_add, list):
        to_add = [to_add]
    to_add = list(filter(lambda item: item != "anonymous", to_add))
    to_add = list(filter(lambda item: item != "unknown", to_add))
    return "by " + separators["caption_inner"].join(to_add)


def to_text_rating(to_add, separators, characters):
    if to_add == "explicit":
        return "explicit"
    else:
        return None


def to_text_tags(to_add, separators, characters):
    # Case of {tag: score}
    if isinstance(to_add, dict):
        to_add = to_add.keys()
    to_add = list(filter(lambda item: item != "unknown", to_add))
    return separators["caption_inner"].join(to_add)


_CAPTIONING_METHODS = {
    "n_people": to_text_npeople,
    "characters": to_text_characters,
    "copyright": to_text_copyright,
    "type": to_text_type,
    "artist": to_text_artist,
    "rating": to_text_rating,
}


def dict_to_caption(info_dict, use_probs, separators, characters):
    caption = ""

    for attribute in _CAPTIONING_METHODS.keys():
        caption = caption_add_content(
            caption,
            info_dict,
            attribute,
            use_probs[attribute],
            _CAPTIONING_METHODS[attribute],
            separators,
            characters,
        )
    if "processed_tags" in info_dict:
        caption = caption_add_content(
            caption,
            info_dict,
            "processed_tags",
            use_probs["tags"],
            to_text_tags,
            separators,
        )
    elif "tags" in info_dict:
        caption = caption_add_content(
            caption, info_dict, "tags", use_probs["tags"], to_text_tags, separators
        )
    return caption
