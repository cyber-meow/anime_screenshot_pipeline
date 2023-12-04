import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Configuration toml files
    parser.add_argument(
        "--base_config_file",
        nargs="?",
        help=(
            "Path to base TOML configuration file. "
            "Configurations from the TOML file override default arguments and "
            "are written by command-line arguments."
        ),
    )
    parser.add_argument(
        "--config_file",
        nargs="*",
        help=(
            "Path to TOML configuration files. "
            "Configurations from the TOML files override default arguments and "
            "are written by command-line arguments."
            "Multiple files can be specified by repeating the argument, in which case "
            "multiple pipelines are executed in parallel."
        ),
    )

    # General arguments
    parser.add_argument(
        "--src_dir",
        type=str,
        default="animes",
        help=(
            "Directory containing source files. "
            "It is ignored if we start from stage 0."
        ),
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="data",
        help="Directory to save output files. It only affects stage 0 to 4.",
    )
    parser.add_argument(
        "--extra_path_component",
        type=str,
        default="",
        help=(
            "Extra path component to add between dst_dir/[training|intermediate] "
            "and image type."
        ),
    )
    parser.add_argument(
        "--start_stage", default="1", help="Stage numbeer or alias to start from"
    )
    parser.add_argument(
        "--end_stage", default="7", help="Stage number or alias to end at"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs. Set to 'None' or 'none' to disable.",
    )
    parser.add_argument(
        "--log_prefix",
        type=str,
        default=None,
        help=(
            "Prefix for log files, "
            "defaults to --anime_name if provided otherwise 'logfile'"
        ),
    )
    parser.add_argument(
        "--overwrite_path",
        action="store_true",
        help=(
            "Overwrite path in metadata. It has effect at stage 2 and 4-7. "
            "Using by starting at stage 4 will prevent from character information "
            "from being written to the original images. "
            "Should never be used in general."
        ),
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default="screenshots",
        choices=["screenshots", "booru"],
        help=(
            "Pipeline type that is used to construct dataset. "
            "Options are 'screenshots' and 'booru'."
        ),
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default=None,
        help=(
            "Image type that we are dealing with, used for folder name. "
            "It may appear in caption if --use_image_type_prob is larger than 0. "
            "Defaults to --pipeline_type."
        ),
    )
    parser.add_argument(
        "--remove_intermediate",
        action="store_true",
        help=(
            "Whether to remove intermediate result or not "
            "(results after stage 1 are always saved)"
        ),
    )

    # Arguments for downloading animes from nyaa.si
    parser.add_argument(
        "--anime_name",
        type=str,
        default=None,
        help="The anime name used for downloading animes from nyaa.si",
    )
    parser.add_argument(
        "--candidate_submitters",
        nargs="+",
        default=["Erai-raws", "SubsPlease", "CameEsp", "ohys"],
        help="The candidate submitters used for downloading animes from nyaa.si",
    )
    parser.add_argument(
        "--anime_resolution",
        type=int,
        default=720,
        help="The resolution of anime to download",
    )
    parser.add_argument(
        "--min_download_episode",
        type=int,
        default=None,
        help="The minimum episode to download",
    )
    parser.add_argument(
        "--max_download_episode",
        type=int,
        default=None,
        help="The maximum episode to download",
    )

    # Arguments for downloading images from Danbooru
    parser.add_argument(
        "--anime_name_booru",
        type=str,
        default=None,
        help=(
            "The anime name used for downloading images from Danbooru. "
            "Defaults to None in which case --anime_name would be used if provided."
        ),
    )
    parser.add_argument(
        "--character_info_file",
        type=str,
        default=None,
        help=(
            "Path to CSV file containing character mapping information "
            "It is used for renaming and potentially for downloading as well"
        ),
    )
    parser.add_argument(
        "--download_for_characters",
        action="store_true",
        help=(
            "Indicates whether to attempt downloading for all characters "
            "in the character info file"
        ),
    )
    parser.add_argument(
        "--booru_download_limit",
        type=int,
        default=None,
        help=(
            "Limit on the total number of images to download from Danbooru. "
            "Defaults to no limit. Setting to 0 will download all images as well. "
            "Note that for efficiency if both --booru_download_limit and "
            "--booru_download_limit_per_character are set, we are not guaranteed "
            "to download --booru_download_limit number of images."
        ),
    )
    parser.add_argument(
        "--booru_download_limit_per_character",
        type=int,
        default=500,
        help=(
            "Limit on the number of images to download per character from Danbooru. "
            "Defaults to 500. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--allowed_ratings",
        nargs="*",
        default=[],
        help=(
            "List of allowed ratings for filtering images. "
            "Should pick from 's', 'g', 'q', 'e'. "
            "Defaults to empty list which means no filtering."
        ),
    )
    parser.add_argument(
        "--allowed_image_classes",
        nargs="*",
        default=["illustration", "bangumi"],
        help=(
            "List of allowed classes for filtering images. "
            "Should pick from 'illustration' 'bangumi' 'comic', and '3d'. "
            "Defauts to ['illustration', 'bangumi']. "
            "Set to empty list to disable."
        ),
    )
    parser.add_argument(
        "--max_download_size",
        type=int,
        default=1024,
        help=(
            "Maximum size for the smaller dimension of too large downloaded images "
            "to resize to"
        ),
    )

    # Arguments for video extraction
    parser.add_argument(
        "--extract_key", action="store_true", help="Only extract key frames"
    )
    parser.add_argument(
        "--image_prefix",
        default="",
        help=(
            "Output image prefix. "
            "Default to None which tries to infer prefix from video name."
        ),
    )
    parser.add_argument(
        "--ep_init",
        type=int,
        default=None,
        help=(
            "Episode number to start with for naming output images. "
            "The processing order of the videos follow the ordering of the names, "
            "so this is not guaranteed to be the same as the true episode order. "
            "Defaults to None which tries to infer episode number from video name."
        ),
    )

    # Arguments for duplicate detection
    parser.add_argument(
        "--no_remove_similar", action="store_true", help="Do not remove similar images"
    )
    parser.add_argument(
        "--detect_duplicate_model",
        default="mobilenetv3_large_100",
        help="Model used for duplicate detection from timm",
    )
    parser.add_argument(
        "--detect_duplicate_batch_size",
        type=int,
        default=16,
        help="Batch size for embendding computation used in duplicate detection",
    )
    parser.add_argument(
        "--similar_thresh",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for image duplicate detection",
    )

    # Arguments for character cropping
    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=320,
        help="Minimum size for character cropping",
    )
    parser.add_argument(
        "--crop_with_head",
        action="store_true",
        help="Crop only images with head for character identification",
    )
    parser.add_argument(
        "--crop_with_face",
        action="store_true",
        help="Crop only images with face for character identification",
    )
    parser.add_argument(
        "--detect_level",
        type=str,
        choices=["n", "s", "m", "x"],
        default="n",
        help=(
            "The detection model level being used. "
            "The 'n' model runs faster with smaller system overhead."
        ),
    )
    parser.add_argument(
        "--use_3stage_crop",
        action="store",
        type=int,
        choices=[2, 4],
        const=2,
        nargs="?",
        help=(
            "Use 3 stage crop to get halfbody and head crops. "
            "This is slow and should only be called once for a set of images. "
            "Possible to use either at stage 2 or 4."
        ),
    )

    # Arguments for character clustering/classification
    # Most important
    parser.add_argument(
        "--character_ref_dir",
        default=None,
        help="Directory containing reference character images",
    )
    parser.add_argument(
        "--n_add_to_ref_per_character",
        type=int,
        default=0,
        help=(
            "The number of additional reference images to add to each character "
            "from classification result"
        ),
    )
    parser.add_argument(
        "--ignore_character_metadata",
        action="store_true",
        help=(
            "Whether to ignore existing character metadata during classification "
            "(only meaning ful for 'booru' pipeline as this is always the case "
            "for 'screenshots' pipeline)"
        ),
    )
    parser.add_argument(
        "--no_extract_from_noise",
        action="store_true",
        help="Whether to disable matching character labels for noise images",
    )
    parser.add_argument(
        "--no_filter_characters",
        action="store_true",
        help="Whether to disable final filtering for character consistency",
    )
    parser.add_argument(
        "--keep_unnamed_clusters",
        action="store_true",
        help=(
            "Whether to keep unnamed clusters when reference images are provided "
            "or when characters are available in metadata"
        ),
    )
    parser.add_argument(
        "--accept_multiple_candidates",
        action="store_true",
        help=(
            "Whether we try to attribute label when multiple candidates are available "
            "when performing classification with metadata character information. "
            "This typically coressponds to the case where we have one character that "
            "always appear with another specific character, or to some specific form "
            "of a character that is recognized as character tag in Danbooru."
        ),
    )
    parser.add_argument(
        "--cluster_merge_threshold",
        type=float,
        default=0.85,
        help="Cluster merge threshold in character clusterining",
    )
    parser.add_argument(
        "--cluster_min_samples",
        type=int,
        default=5,
        help="Minimum cluster samples in character clusterining",
    )
    parser.add_argument(
        "--same_threshold_rel",
        type=float,
        default=0.6,
        help=(
            "The relative threshold for determining whether images belong to "
            "the same cluster for noise extraction and filtering"
        ),
    )
    parser.add_argument(
        "--same_threshold_abs",
        type=int,
        default=20,
        help=(
            "The absolute threshold for determining whether images belong to "
            "the same cluster for noise extraction and filtering"
        ),
    )

    # Arguments for dataset construction
    parser.add_argument(
        "--overwrite_emb_init_info",
        action="store_true",
        help="Overwrite existing trigger word csv",
    )
    parser.add_argument(
        "--character_overwrite_uncropped",
        action="store_true",
        help=(
            "Overwrite existing character metadata for uncropped images "
            "(only meaningful for 'booru' pipeline as this is always the case "
            "for 'screenshots' pipeline)"
        ),
    )
    parser.add_argument(
        "--character_remove_unclassified",
        action="store_true",
        help=(
            "Remove unclassified characters in the character metadata field "
            "(only has effect for 'booru' pipeline without "
            "--character_overwrite_uncropped)"
        ),
    )
    parser.add_argument(
        "--no_resize", action="store_true", help="Do not perform image resizing"
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=768,
        help="Max image size that shorter edge aligns to",
    )
    parser.add_argument(
        "--image_save_ext", default=".webp", help="Dataset image extension"
    )
    parser.add_argument(
        "--n_anime_reg",
        type=int,
        default=500,
        help="Number of images with no characters to keep (for 'screenshots' pipeline)",
    )
    parser.add_argument(
        "--filter_again", action="store_true", help="Filter repeated images again here"
    )

    # Loading and saving of metadata for tagging and captioning stage
    parser.add_argument(
        "--load_aux",
        type=str,
        nargs="*",
        # default=['processed_tags', 'characters'],
        help="List of auxiliary attributes to load. "
        # "Default is processed_tags and characters. "
        "E.g., --load_aux attr1 attr2 attr3",
    )
    parser.add_argument(
        "--save_aux",
        type=str,
        nargs="*",
        # default=['processed_tags', 'characters'],
        help="List of auxiliary attributes to save. "
        # "Default is processed_tags and characters. "
        "E.g., --save_aux attr1 attr2 attr3",
    )

    # Arguments for tagging
    parser.add_argument(
        "--overwrite_tags",
        action="store_true",
        help="Whether to overwrite existing tags",
    )
    parser.add_argument(
        "--tagging_method",
        type=str,
        default="wd14_convnextv2",
        help=(
            "Method used for tagging. "
            "Options are 'deepdanbooru', 'wd14_vit', 'wd14_convnext', "
            "'wd14_convnextv2', 'wd14_swinv2', 'mldanbooru'."
        ),
    )
    parser.add_argument(
        "--tag_threshold", type=float, default=0.35, help="Threshold for tagging"
    )

    # General arguments for tag processing
    parser.add_argument(
        "--sort_mode",
        type=str,
        default="score",
        choices=["score", "shuffle", "original"],
        help=(
            "Mode to sort the tags. "
            "Options are 'score', 'shuffle', 'original'. "
            "Default is 'score'."
        ),
    )
    parser.add_argument(
        "--append_dropped_character_tags",
        action="store_true",
        help="Whether to append dropped character tags",
    )
    parser.add_argument(
        "--max_tag_number",
        type=int,
        default=30,
        help="Max number of tags to include in caption",
    )
    parser.add_argument(
        "--blacklist_tags_file",
        type=str,
        default="configs/tag_filtering/blacklist_tags.txt",
        help="Path to the file containing blacklisted tags",
    )
    parser.add_argument(
        "--overlap_tags_file",
        type=str,
        default="configs/tag_filtering/overlap_tags.json",
        help="Path to the file containing overlap tag information",
    )
    parser.add_argument(
        "--character_tags_file",
        type=str,
        default="configs/tag_filtering/character_tags.json",
        help="Path to the file containing character tag information",
    )
    parser.add_argument(
        "--process_from_original_tags",
        action="store_true",
        help="process tags from original tags instead of processed tags",
    )
    parser.add_argument(
        "--prune_mode",
        type=str,
        default="character_core",
        choices=["character", "character_core", "minimal", "none"],
        help=(
            "Different ways to prune tags. "
            "Options are 'character', 'character_core', 'minimal', 'none'. "
            "Default is 'character_core'."
        ),
    )

    # Specific arguments for core tag processing
    parser.add_argument(
        "--compute_core_tag_up_levels",
        type=int,
        default=1,
        help=(
            "Number of directory levels to go up from the captioned directory when "
            "computing core tags. Defaults to 1 which computes for all the image types."
        ),
    )
    parser.add_argument(
        "--core_frequency_thresh",
        type=float,
        default=0.4,
        help="Minimum frequency for a tag to be considered core tag.",
    )
    parser.add_argument(
        "--use_existing_core_tag_file",
        action="store_true",
        help="Use existing core tag json instead of recomputing them.",
    )
    parser.add_argument(
        "--drop_difficulty",
        type=int,
        default=2,
        help=(
            "The difficulty level up to which tags should be dropped. Tags with "
            "difficulty less than this value will be added to the drop lists. "
            "0: nothing is dropped; 1: human-related tag; 2: furry, demon, mecha, etc."
            "Defaults to 2."
        ),
    )
    parser.add_argument(
        "--drop_all_core",
        action="store_true",
        help=("Whether to drop all core tags or not. Overwrites --drop_difficulty."),
    )
    parser.add_argument(
        "--emb_min_difficulty",
        type=int,
        default=1,
        help=(
            "The difficulty level from which tags should be used for embedding "
            "initialization. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--emb_max_difficulty",
        type=int,
        default=2,
        help=(
            "The difficulty level up to which tags should be used for embedding "
            "initialization. Defaults to 2."
        ),
    )
    parser.add_argument(
        "--emb_init_all_core",
        action="store_true",
        help=(
            "Whether to use all core tags for embedding initialization. "
            "Overwrites --emb_min_difficulty and --emb_max_difficulty."
        ),
    )
    parser.add_argument(
        "--append_dropped_character_tags_wildcard",
        action="store_true",
        help="Whether to append dropped character tags to wildcard",
    )

    # Arguments for captioning
    parser.add_argument(
        "--caption_ordering",
        nargs="*",
        default=[
            "npeople",
            "character",
            "copyright",
            "image_type",
            "artist",
            "rating",
            "tags",
        ],
        help=(
            "For specifying the order of captions. "
            "Defaults to ['npeople', 'character', 'copyright', "
            "'image_type', 'artist', 'rating']"
        ),
    )
    parser.add_argument(
        "--caption_inner_sep",
        type=str,
        default=", ",
        help="For separating items of a single field of caption",
    )
    parser.add_argument(
        "--caption_outer_sep",
        type=str,
        default=", ",
        help="For separating different fields of caption",
    )
    parser.add_argument(
        "--character_sep",
        type=str,
        default=", ",
        help="For separating characters",
    )
    parser.add_argument(
        "--character_inner_sep",
        type=str,
        default=" ",
        help="For separating items of a single field of character",
    )
    parser.add_argument(
        "--character_outer_sep",
        type=str,
        default=", ",
        help="For separating different fields of character",
    )
    parser.add_argument(
        "--keep_tokens_sep",
        type=str,
        default=None,
        help=(
            "Separator for keep tokens for Kohya trainer. "
            "Defaults to None which uses --character_outer_sep."
        ),
    )
    parser.add_argument(
        "--keep_tokens_before",
        type=str,
        default="tags",
        help="Where to put --keep_token_sep before. Defaults to 'tags'.",
    )
    parser.add_argument(
        "--use_npeople_prob",
        type=float,
        default=0,
        help=(
            "Probability to include number of people in captions. "
            "Including number of people in captions however is not yet implemented."
        ),
    )
    parser.add_argument(
        "--use_character_prob",
        type=float,
        default=1,
        help="Probability to include character info in captions",
    )
    parser.add_argument(
        "--use_copyright_prob",
        type=float,
        default=0,
        help="Probability to include copyright info in captions",
    )
    parser.add_argument(
        "--use_image_type_prob",
        type=float,
        default=1,
        help="Probability to include image type info in captions",
    )
    parser.add_argument(
        "--use_artist_prob",
        type=float,
        default=0,
        help="Probability to include artist info in captions",
    )
    parser.add_argument(
        "--use_rating_prob",
        type=float,
        default=1,
        help="Probability to include rating info in captions",
    )
    parser.add_argument(
        "--use_tags_prob",
        type=float,
        default=1,
        help="Probability to include tag info in captions",
    )

    # Arguments for folder organization
    parser.add_argument(
        "--rearrange_up_levels",
        type=int,
        default=0,
        help=(
            "Number of directory levels to go up from the captioned directory when "
            "setting the source directory for the rearrange stage."
            "Defaults to 0."
        ),
    )
    parser.add_argument(
        "--arrange_format",
        type=str,
        default="n_characters/character",
        help="Description of the concept balancing directory hierarchy",
    )
    parser.add_argument(
        "--max_character_number",
        type=int,
        default=6,
        help="If have more than X characters put X+",
    )
    parser.add_argument(
        "--min_images_per_combination",
        type=int,
        default=10,
        help=(
            "Put others instead of character name if number of images "
            "of the character combination is smaller then this number"
        ),
    )

    # For dataset balancing
    parser.add_argument(
        "--compute_multiply_up_levels",
        type=int,
        default=1,
        help=(
            "Number of directory levels to go up from the rearranged directory when "
            "setting the source directory for the compute multiply stage. "
            "Defaults to 1."
        ),
    )
    parser.add_argument(
        "--min_multiply", type=float, default=1, help="Minimum multiply of each image"
    )
    parser.add_argument(
        "--max_multiply", type=int, default=100, help="Maximum multiply of each image"
    )
    parser.add_argument(
        "--weight_csv",
        default="configs/csv_examples/default_weighting.csv",
        help="If provided use the provided csv to modify weights",
    )

    args = parser.parse_args()

    explicit_args = {
        key: value for key, value in vars(args).items() if sys.argv.count("--" + key)
    }
    return args, explicit_args
