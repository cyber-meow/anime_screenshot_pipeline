"""Check embedding names are valid and update embedding information
with initialization text
"""
import os
import json
import logging
from typing import Optional, List, Dict
from transformers import AutoTokenizer


def update_emb_init_info(
    filepath: str,
    characters: List[str],
    image_type: str,
    emb_init_dict: Optional[Dict[str, List[str]]] = None,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Updates the JSON file with character names and optionally with embedding
    initialization information.
    Additionally, checks the validity of embedding names for HCP training.

    Args:
        filepath (str):
            Path to the trigger word JSON file.
        characters (List[str]):
            List of character names to add.
        image_type (str):
            Type of the image ("screenshots", "booru", or other).
        emb_init_dict (Optional[Dict[str, List[str]]]):
            Optional dictionary for embedding initializations.
        overwrite (bool):
            Whether to overwrite existing JSON content.
        logger (Optional[logging.Logger]):
            Optional logger to use. Defaults to None, which uses the default logger.
    """
    if logger is None:
        logger = logging.getLogger()
    name_init_map = {}

    # Read existing content if not overwriting
    if not overwrite and os.path.exists(filepath):
        with open(filepath, "r") as file:
            name_init_map = json.load(file)

    # Add characters to the map
    for character in characters:
        embedding_name = character.split()[0]
        if embedding_name not in name_init_map:
            name_init_map[embedding_name] = []

    # Add image_type to the map
    if image_type not in name_init_map:
        if image_type == "screenshots":
            default_init_text = "anime screencap"
        elif image_type == "booru":
            default_init_text = "masterpiece"
        else:
            default_init_text = ""
        name_init_map[image_type] = [default_init_text]

    # Update with emb_init_dict
    if emb_init_dict:
        for emb, tags in emb_init_dict.items():
            if emb in name_init_map:
                # Add new tags to the existing list, avoiding duplicates
                name_init_map[emb].extend(
                    [tag for tag in tags if tag not in name_init_map[emb]]
                )
            else:
                name_init_map[emb] = tags

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    invalid_embedding_names = []
    for embedding_name in name_init_map.keys():
        if embedding_name.lower() in tokenizer.vocab:
            invalid_embedding_names.append(embedding_name)

    # Log warning for invalid embedding names
    if invalid_embedding_names:
        logger.warning(
            "Some embedding names may not be valid for HCP training: "
            + ", ".join(invalid_embedding_names)
        )

    # Write the updated content back to the JSON file
    with open(filepath, "w") as file:
        json.dump(name_init_map, file, indent=4)
