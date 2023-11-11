import random


def caption_add_content(
    caption,
    info_dict,
    attribute,
    prob,
    to_text,
    sep_string=", ",
    characters=None,
):
    if random.random() >= prob or attribute not in info_dict:
        return caption
    to_add = info_dict[attribute]
    if not to_add:
        return caption
    to_add_text = to_text(to_add, sep_string, characters)
    if to_add_text is not None:
        if caption != "":
            caption += sep_string
        caption += to_add_text
    return caption


def to_text_npeople(count, sep_string, characters):
    suffix = "person" if count == 1 else "people"
    return f"{count}{suffix}"


def to_text_characters(to_add, sep_string, characters):
    if not isinstance(to_add, list):
        to_add = [to_add]
    # Ideally this should be filtered in earlier stages
    # Still left here just in case
    if characters is None:
        to_add = list(filter(lambda item: item != "unknown", to_add))
    else:
        to_add = list(filter(lambda item: item in characters, to_add))
    return sep_string.join(to_add)


def to_text_copyright(to_add, sep_sttring, characters):
    if not isinstance(to_add, list):
        to_add = [to_add]
    to_add = list(filter(lambda item: item != "unknown", to_add))
    return "from " + ", ".join(to_add)


def to_text_type(image_type, sep_string, characters):
    return image_type


def to_text_artist(to_add, sep_string, characters):
    if not isinstance(to_add, list):
        to_add = [to_add]
    to_add = list(filter(lambda item: item != "anonymous", to_add))
    to_add = list(filter(lambda item: item != "unknown", to_add))
    return "by " + ", ".join(to_add)


def to_text_rating(to_add, sep_string, characters):
    if to_add == "explicit":
        return "explicit"
    else:
        return None


def to_text_tags(to_add, sep_string, characters):
    # Case of {tag: score}
    if isinstance(to_add, dict):
        to_add = to_add.keys()
    to_add = list(filter(lambda item: item != "unknown", to_add))
    return ", ".join(to_add)


_CAPTIONING_METHODS = {
    "n_people": to_text_npeople,
    "characters": to_text_characters,
    "copyright": to_text_copyright,
    "type": to_text_type,
    "artist": to_text_artist,
    "rating": to_text_rating,
}


def dict_to_caption(info_dict, args, characters):
    caption = ""
    sep_string = args.caption_separator + " "
    use_probs = [
        args.use_npeople_prob,
        args.use_character_prob,
        args.use_copyright_prob,
        args.use_image_type_prob,
        args.use_artist_prob,
        args.use_rating_prob,
    ]

    for attribute, prob in zip(_CAPTIONING_METHODS.keys(), use_probs):
        caption = caption_add_content(
            caption,
            info_dict,
            attribute,
            prob,
            _CAPTIONING_METHODS[attribute],
            sep_string,
            characters,
        )
    if "processed_tags" in info_dict:
        caption = caption_add_content(
            caption,
            info_dict,
            "processed_tags",
            args.use_tags_prob,
            to_text_tags,
            sep_string,
        )
    elif "tags" in info_dict:
        caption = caption_add_content(
            caption, info_dict, "tags", args.use_tags_prob, to_text_tags, sep_string
        )
    return caption
