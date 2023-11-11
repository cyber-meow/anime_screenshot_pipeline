import argparse
from anime2sd import get_character_core_tags_and_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frequent tags for characters.")
    parser.add_argument(
        "--src_dir", type=str,
        help="Path to the folder containing images and metadata.")
    parser.add_argument(
        "--frequency_threshold", type=float, default=0.5,
        help="Minimum frequency for a tag to be considered core tag.")
    parser.add_argument(
        "--core_tag_output", type=str, default="core_tags.json",
        help="Output JSON file to save the frequent tags.")
    parser.add_argument(
        "--wildcard_output", type=str, default="wildcard.txt",
        help="Output TXT file to save the character names and their tags.")

    args = parser.parse_args()
    get_character_core_tags_and_save(
        args.src_dir, args.core_tag_output, args.wildcard_output,
        frequency_threshold=args.frequency_threshold)
