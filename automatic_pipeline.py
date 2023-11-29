import os
import sys
import toml
import copy
import shutil
import logging
import argparse
from datetime import datetime

from waifuc.action import PersonSplitAction
from waifuc.action import MinSizeFilterAction
from waifuc.action import ThreeStageSplitAction

from anime2sd import extract_and_remove_similar
from anime2sd import classify_from_directory
from anime2sd import select_dataset_images_from_directory
from anime2sd import tag_and_caption_from_directory
from anime2sd import arrange_folder, get_repeat
from anime2sd import read_weight_mapping
from anime2sd import rearrange_related_files, load_metadata_from_aux
from anime2sd import CharacterTagProcessor, TaggingManager, CaptionGenerator

from anime2sd.waifuc_customize import LocalSource, SaveExporter
from anime2sd.waifuc_customize import MinFaceCountAction, MinHeadCountAction


def update_args_from_toml(
    args: argparse.Namespace, toml_path: str
) -> argparse.Namespace:
    """
    Update a copy of args with configurations from a TOML file.

    This function reads a TOML file and updates the attributes of the given
    argparse.Namespace object with the configurations found in the file.
    If the TOML file contains nested sections, they are flattened.

    Args:
        args (argparse.Namespace):
            The original argparse Namespace object containing command-line arguments.
        toml_path (str):
            Path to the TOML configuration file.

    Returns:
        argparse.Namespace:
            A new Namespace object with updated configurations from the TOML file.

    Raises:
        Exception: If there is an error in reading or parsing the TOML file.
    """
    new_args = copy.deepcopy(args)
    try:
        with open(toml_path, "r") as f:
            config = toml.load(f)
        for key, value in config.items():
            if isinstance(value, dict):
                # Handle nested sections by flattening them
                for nested_key, nested_value in value.items():
                    setattr(new_args, nested_key, nested_value)
            else:
                setattr(new_args, key, value)
    except Exception as e:
        print(f"Error loading config from {toml_path}: {e}")
    return new_args


def setup_logging(log_dir: str, log_prefix: str):
    """
    Set up logging to file and stdout with specified directory and prefix.

    Args:
        log_dir (str): Directory to save the log file.
        log_prefix (str): Prefix for the log file name.
    """

    # Create logger
    logger = logging.getLogger()
    original_handlers = logger.handlers[:]
    for handler in original_handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Add formatter to ch
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Create file handler and set level to info
    if log_dir.lower() != "none":
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now()
        str_current_time = current_time.strftime("%Y-%m-%d%H-%M-%S")
        log_file = os.path.join(log_dir, f"{log_prefix}_{str_current_time}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        fh.setFormatter(formatter)


def setup_args(args):
    """
    Sets up the start and end stages for the pipeline based on the provided arguments.
    If the 'image_type' is not specified, it defaults to the 'pipeline_type'.
    """
    # Mapping stage numbers to their aliases
    STAGE_ALIASES = {
        1: ["extract"],
        2: ["crop"],
        3: ["classify"],
        4: ["select"],
        5: ["tag", "caption", "tag_and_caption"],
        6: ["arrange"],
        7: ["balance"],
    }
    if not config.image_type:
        config.image_type = args.pipeline_type

    start_stage = args.start_stage
    end_stage = args.end_stage

    # Convert stage aliases to numbers if provided
    for stage_number in STAGE_ALIASES:
        if args.start_stage in STAGE_ALIASES[stage_number]:
            start_stage = stage_number
        if args.end_stage in STAGE_ALIASES[stage_number]:
            end_stage = stage_number

    args.start_stage = int(start_stage)
    args.end_stage = int(end_stage)


def get_and_create_dst_dir(
    args: argparse.Namespace,
    mode: str,
    sub_dir: str = "",
    makedirs: bool = True,
) -> str:
    """
    Constructs the destination directory path based on the mode, subdirectory,
    and additional arguments.

    If 'makedirs' is True, the function also creates the directory if it doesn't exist.

    Args:
        args (argparse.Namespace):
            The namespace object containing the command-line arguments.
        mode (str):
            The mode specifying the main directory under the destination directory.
        sub_dir (str, optional):
            An additional subdirectory to put at the end.
            Defaults to an empty string.
        makedirs (bool, optional):
            Whether to create the directory if it doesn't exist. Defaults to True.

    Returns:
        str: The path to the constructed destination directory.
    """
    dst_dir = os.path.join(
        args.dst_dir, mode, args.extra_path_component, args.image_type, sub_dir
    )
    if makedirs:
        os.makedirs(dst_dir, exist_ok=True)
    return dst_dir


def get_src_dir(args, stage):
    """
    Determines the source directory for a given stage of the pipeline.

    Args:
        args (argparse.Namespace):
            The namespace object containing the command-line arguments.
        stage (int): The current stage of the pipeline.

    Returns:
        str: The path to the source directory for the given stage.

    Raises:
        ValueError: If the provided stage number is invalid.
    """
    if stage == args.start_stage or stage == 1:
        return args.src_dir
    elif stage == 2:
        return get_and_create_dst_dir(args, "intermediate", "raw", makedirs=False)
    elif stage == 3:
        return get_and_create_dst_dir(args, "intermediate", "cropped", makedirs=False)
    elif stage == 4:
        return get_and_create_dst_dir(args, "intermediate", makedirs=False)
    elif stage == 5:
        return get_and_create_dst_dir(args, "training", makedirs=False)
    elif stage == 6:
        dst_dir = get_src_dir(args, 5)
        for _ in range(args.rearrange_up_levels):
            dst_dir = os.path.dirname(dst_dir)
        return dst_dir
    elif stage == 7:
        dst_dir = get_src_dir(args, 6)
        for _ in range(args.compute_multiply_up_levels):
            dst_dir = os.path.dirname(dst_dir)
        return dst_dir
    else:
        raise ValueError(f"Invalid stage: {stage}")


def extract_frames(args, stage):
    """
    Extracts frames from videos and saves them to the destination directory.
    This function also handles duplicate detection and removal.
    """
    # Get the path to the source directory containing the videos
    src_dir = get_src_dir(args, stage)
    # Get the path to the destination directory for the extracted frames
    dst_dir = get_and_create_dst_dir(args, "intermediate", "raw")
    logging.info(f"Extracting frames to {dst_dir} ...")

    extract_and_remove_similar(
        src_dir,
        dst_dir,
        args.image_prefix,
        ep_init=args.ep_init,
        extract_key=args.extract_key,
        model_name=args.detect_duplicate_model,
        thresh=args.similar_thresh,
        to_remove_similar=not args.no_remove_similar,
    )


# TODO: Avoid cropping for already cropped data
def crop_characters(args, stage):
    """Crops individual characters from images in the source directory."""
    # Get the path to the source directory containing the images to crop from
    src_dir = get_src_dir(args, stage)
    # Get the path to the destination directory for the cropped images
    dst_dir = get_and_create_dst_dir(args, "intermediate", "cropped")
    logging.info(f"Cropping individual characters to {dst_dir} ...")

    overwrite_path = args.start_stage == stage and args.overwrite_path

    source = LocalSource(src_dir, overwrite_path=overwrite_path)
    detect_config_person = {"level": args.detect_level}
    if args.detect_level in ["s", "n"]:
        detect_level_head_halfbody = args.detect_level
    else:
        detect_level_head_halfbody = "n"
    detect_config = {"level": detect_level_head_halfbody}
    crop_action = (
        ThreeStageSplitAction(
            split_person=True,
            head_conf=detect_config,
            halfbody_conf=detect_config,
            person_conf=detect_config_person,
        )
        if args.use_3stage_crop == 2
        else PersonSplitAction(keep_original=False, level=args.detect_level)
    )

    source = source.attach(
        # NoMonochromeAction(),
        crop_action,
        MinSizeFilterAction(args.min_crop_size),
        # Not used here because it can be problematic for multi-character scene
        # Some not moving while other moving
        # FilterSimilarAction('all'),
    )
    if args.crop_with_head:
        source = source.attach(
            MinHeadCountAction(1, level="n"),
        )
    if args.crop_with_face:
        source = source.attach(
            MinFaceCountAction(1, level="n"),
        )

    source.export(SaveExporter(dst_dir, no_meta=False, save_caption=False))


def classify_characters(args, stage):
    """Classifies characters in the given source directory."""

    # Get the path to the source directory containing images to be classified
    src_dir = get_src_dir(args, stage)
    # Get the path to the distination directory containing the classified images
    dst_dir = get_and_create_dst_dir(args, "intermediate", "classified")

    # Determine whether to move or copy files to the destination directory.
    move = args.remove_intermediate or (src_dir == dst_dir)
    # Determine whether to ignore existing character metadata.
    ignore_character_metadata = (
        args.ignore_character_metadata or args.pipeline_type == "screenshots"
    )

    # Log information about the classification process.
    logging.info(f"Classifying characters to {dst_dir} ...")

    # Call the `classify_from_directory` function with the specified parameters.
    classify_from_directory(
        src_dir,
        dst_dir,
        ref_dir=args.character_ref_dir,
        ignore_character_metadata=ignore_character_metadata,
        to_extract_from_noise=not args.no_extract_from_noise,
        to_filter=not args.no_filter_characters,
        keep_unnamed=args.keep_unnamed_clusters,
        clu_min_samples=args.cluster_min_samples,
        merge_threshold=args.cluster_merge_threshold,
        same_threshold_rel=args.same_threshold_rel,
        same_threshold_abs=args.same_threshold_abs,
        move=move,
    )


def select_dataset_images(args, stage):
    """Construct training set from classified images and raw images."""
    # Get the path to the intermediate directory containing the
    # two folders "raw" and "classified".
    src_dir = get_src_dir(args, stage)
    classified_dir = os.path.join(src_dir, "classified")
    full_dir = os.path.join(src_dir, "raw")
    # Get the path to the image_type subfolder of the training directory
    dst_dir = get_and_create_dst_dir(args, "training")

    is_start_stage = args.start_stage == stage
    if is_start_stage:
        # rearrange json and ccip in case of manual inspection
        rearrange_related_files(classified_dir)
    overwrite_path = is_start_stage and args.overwrite_path

    logging.info(f"Preparing dataset images to {dst_dir} ...")

    select_dataset_images_from_directory(
        classified_dir,
        full_dir,
        dst_dir,
        pipeline_type=args.pipeline_type,
        overwrite_path=overwrite_path,
        # For saving character to metadata
        character_overwrite_uncropped=args.character_overwrite_uncropped,
        character_remove_unclassified=args.character_remove_unclassified,
        # For saving embedding initialization information
        image_type=args.image_type,
        overwrite_emb_init_info=args.overwrite_emb_init_info,
        # For 3 stage cropping
        use_3stage_crop=args.use_3stage_crop == 4,
        detect_level=args.detect_level,
        # For resizing/copying images to destination
        max_size=args.max_size,
        image_save_ext=args.image_save_ext,
        to_resize=not args.no_resize,
        n_anime_reg=args.n_anime_reg,
        # For additional filtering after obtaining dataset images
        filter_again=args.filter_again,
        detect_duplicate_model=args.detect_duplicate_model,
        similarity_threshold=args.similar_thresh,
    )

    if args.remove_intermediate:
        shutil.rmtree(classified_dir)


def tag_and_caption(args, stage):
    """Perform in-place tagging and captioning."""
    # Get path to the directiry containing images to be tagged and captioned
    src_dir = get_src_dir(args, stage)
    if args.start_stage == stage:
        # rearrange json and ccip in case of manual inspection
        rearrange_related_files(src_dir)

    if "character" in args.pruned_mode:
        char_tag_proc = CharacterTagProcessor(
            tag_list_path=args.character_tags_file,
            drop_difficulty=args.drop_difficulty,
            emb_min_difficulty=args.emb_min_difficulty,
            emb_max_difficutly=args.emb_max_difficulty,
            drop_all=args.drop_all_core,
            emb_init_all=args.emb_init_all_core,
        )
    else:
        char_tag_proc = None

    tagging_manager = TaggingManager(
        tagging_method=args.tagging_method,
        tag_threshold=args.tag_threshold,
        overwrite_tags=args.overwrite_tags,
        pruned_mode=args.pruned_mode,
        blacklist_tags_file=args.blacklist_tags_file,
        overlap_tags_file=args.overlap_tags_file,
        character_tag_processor=char_tag_proc,
        process_from_original_tags=args.process_from_original_tags,
        sort_mode=args.sort_mode,
        max_tag_number=args.max_tag_number,
    )

    caption_generator = CaptionGenerator(
        character_sep=args.character_sep,
        character_inner_sep=args.character_inner_sep,
        character_outer_sep=args.character_outer_sep,
        caption_inner_sep=args.caption_inner_sep,
        caption_outer_sep=args.caption_outer_sep,
        use_npeople_prob=args.use_npeople_prob,
        use_character_prob=args.use_character_prob,
        use_copyright_prob=args.use_copyright_prob,
        use_image_type_prob=args.use_image_type_prob,
        use_artist_prob=args.use_artist_prob,
        use_rating_prob=args.use_rating_prob,
        use_tags_prob=args.use_tags_prob,
    )

    logging.info(f"Tagging and captioning images in {src_dir} ...")

    tag_and_caption_from_directory(
        src_dir,
        tagging_manager,
        caption_generator,
        # For core tags
        use_existing_core_tag_file=args.use_existing_core_tag_file,
        core_frequency_threshold=args.core_frequency_thresh,
        # For saving embedding initialization information
        image_type=args.image_type,
        overwrite_emb_init_info=args.overwrite_emb_init_info,
        # For file io
        load_aux=args.load_aux,
        save_aux=args.save_aux,
        overwrite_path=args.overwrite_path,
    )


def rearrange(args, stage):
    """Rearrange the images in the directory."""
    # Get path to the directiry containing images to be rearranged
    src_dir = get_src_dir(args, stage)
    logging.info(f"Rearranging {src_dir} ...")
    if args.start_stage == stage and args.load_aux:
        load_metadata_from_aux(
            src_dir, args.load_aux, args.save_aux, args.overwrite_path
        )
        rearrange_related_files(src_dir)
    arrange_folder(
        src_dir,
        src_dir,
        args.arrange_format,
        args.max_character_number,
        args.min_images_per_combination,
    )


def balance(args, stage):
    """Compute the repeat for the images in the directory."""
    # Get path to the directiry containing images for which repeat needs to be computed
    src_dir = get_src_dir(args, stage)
    if args.start_stage == stage and args.load_aux:
        load_metadata_from_aux(
            src_dir, args.load_aux, args.save_aux, args.overwrite_path
        )
        rearrange_related_files(src_dir)
    logging.info(f"Computing repeat for {src_dir} ...")
    if args.weight_csv is not None:
        weight_mapping = read_weight_mapping(args.weight_csv)
    else:
        weight_mapping = None
    current_time = datetime.now()
    str_current_time = current_time.strftime("%Y-%m-%d%H-%M-%S")
    if args.log_dir.lower() == "none":
        log_file = None
    else:
        log_file = os.path.join(
            args.log_dir, f"{args.log_prefix}_weighting_{str_current_time}.log"
        )
    get_repeat(src_dir, weight_mapping, args.min_multiply, args.max_multiply, log_file)


def launch_pipeline(args):
    # Mapping stage numbers to their respective function names
    STAGE_FUNCTIONS = {
        1: extract_frames,
        2: crop_characters,
        3: classify_characters,
        4: select_dataset_images,
        5: tag_and_caption,
        6: rearrange,
        7: balance,
    }

    setup_logging(args.log_dir, f"{args.pipeline_type}_{args.log_prefix}")

    # Loop through the stages and execute them
    for stage_num in range(args.start_stage, args.end_stage + 1):
        logging.info(f"-------------Start stage {stage_num}-------------")
        STAGE_FUNCTIONS[stage_num](args, stage_num)


if __name__ == "__main__":
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
        help="Directory containing source files",
    )
    parser.add_argument(
        "--dst_dir", type=str, default="data", help="Directory to save output files"
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
        help="Directory to save logs. Set to None or none to disable.",
    )
    parser.add_argument(
        "--log_prefix", type=str, default="logfile", help="Prefix for log files"
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
            "Pipeline type that is used to construct dataset. ",
            "Options are 'screenshots' and 'booru'.",
        ),
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default=None,
        help=(
            "Image type that we are dealing with, used for folder name. "
            "It may appear in caption if 'use_image_type_prob' is larger than 0. "
            "Defaults to pipeline_type."
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

    # Arguments for video extraction
    parser.add_argument(
        "--extract_key", action="store_true", help="Only extract key frames"
    )
    parser.add_argument("--image_prefix", default="", help="Output image prefix")
    parser.add_argument(
        "--ep_init", type=int, default=1, help="Episode number to start with"
    )

    # Arguments for duplicate detection
    parser.add_argument(
        "--no_remove_similar", action="store_true", help="Do not remove similar images"
    )
    parser.add_argument(
        "--detect_duplicate_model",
        default="mobilenet-v2-imagenet-torch",
        help="Model used for duplicate detection",
    )
    parser.add_argument(
        "--similar_thresh",
        type=float,
        default=0.96,
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
        "--ignore_character_metadata",
        action="store_true",
        help=(
            "Whether to ignore existing character metadata during classification ",
            "(only meaning ful for 'booru' pipeline as this is always the case "
            "for 'screenshots' pipeline)",
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
        help="Whether to overwrite existing tags.",
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
        "--tag_threshold", type=float, default=0.35, help="Threshold for tagging."
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
        "--max_tag_number",
        type=int,
        default=30,
        help="Max number of tags to include in caption.",
    )
    parser.add_argument(
        "--blacklist_tags_file",
        type=str,
        default="configs/tag_filtering/blacklist_tags.txt",
        help="Path to the file containing blacklisted tags.",
    )
    parser.add_argument(
        "--overlap_tags_file",
        type=str,
        default="configs/tag_filtering/overlap_tags.json",
        help="Path to the file containing overlap tag information.",
    )
    parser.add_argument(
        "--character_tags_file",
        type=str,
        default="configs/tag_filtering/character_tags.json",
        help="Path to the file containing character tag information.",
    )
    parser.add_argument(
        "--process_from_original_tags",
        action="store_true",
        help="process tags from original tags instead of processed tags",
    )
    parser.add_argument(
        "--pruned_mode",
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

    # Arguments for captioning
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
        "--use_npeople_prob",
        type=float,
        default=0,
        help="Probability to include number of people in captions",
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

    if args.base_config_file:
        args = update_args_from_toml(args, args.base_config_file)

    configs = []
    if args.config_file:
        for toml_path in args.config_file:
            config_args = update_args_from_toml(args, toml_path)
            configs.append(config_args)
    else:
        configs.append(args)

    if args.base_config_file or args.config_file:
        # Overwrite args with explicitly set command line arguments
        for config in configs:
            for key, value in explicit_args.items():
                setattr(config, key, value)

    # A set to record dst_dir and image_type in configs
    dst_folder_set = set()

    # Process each configuration
    for config in configs:
        setup_args(config)
        dst_folder = (config.dst_dir, config.extra_path_component, config.image_type)
        if dst_folder in dst_folder_set:
            raise ValueError(
                "Duplicate (dst_dir, extra_path_component, image_type) "
                "is not supported: "
                f"{config.dst_dir}, {config.extra_path_component}, {config.image_type}"
            )
        dst_folder_set.add(dst_folder)
        launch_pipeline(config)
