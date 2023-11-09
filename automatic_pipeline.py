import os
import json
import shutil
import logging
import argparse
from datetime import datetime

import fiftyone.zoo as foz

from waifuc.action import PersonSplitAction
from waifuc.action import MinSizeFilterAction
from waifuc.action import TaggingAction
from waifuc.action import ThreeStageSplitAction

from anime2sd import extract_and_remove_similar, remove_similar_from_dir
from anime2sd import classify_from_directory
from anime2sd import rearrange_related_files
from anime2sd import save_characters_to_meta, update_trigger_word_info
from anime2sd import resize_character_images
from anime2sd import parse_overlap_tags, read_weight_mapping
from anime2sd import CharacterTagProcessor
from anime2sd import get_character_core_tags, get_character_core_tags_and_save
from anime2sd import save_core_tag_info
from anime2sd import arrange_folder, get_repeat

from anime2sd.waifuc_customize import LocalSource, SaveExporter
from anime2sd.waifuc_customize import TagPruningAction, TagSortingAction
from anime2sd.waifuc_customize import CoreCharacterTagPruningAction
from anime2sd.waifuc_customize import TagRemovingUnderscoreAction
from anime2sd.waifuc_customize import CaptioningAction
from anime2sd.waifuc_customize import MinFaceCountAction, MinHeadCountAction


def setup_logging(log_dir, log_prefix):
    """
    Set up logging to file and stdout with specified directory and prefix.

    Args:
        log_dir: Directory to save the log file.
        log_prefix: Prefix for the log file name.

    Returns: None
    """

    # Create logger
    logger = logging.getLogger()
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


def extract_frames(args, src_dir, is_start_stage):
    dst_dir = os.path.join(args.dst_dir, "intermediate", args.image_type, "raw")
    os.makedirs(dst_dir, exist_ok=True)
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
    return dst_dir


def crop_characters(args, src_dir, is_start_stage):
    overwrite_path = is_start_stage and args.overwrite_path
    source = LocalSource(src_dir, overwrite_path=overwrite_path)
    source = source.attach(
        # NoMonochromeAction(),
        PersonSplitAction(keep_original=False, level="n"),
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

    dst_dir = os.path.join(args.dst_dir, "intermediate", args.image_type, "cropped")
    os.makedirs(dst_dir, exist_ok=True)
    logging.info(f"Cropping individual characters to {dst_dir} ...")
    source.export(SaveExporter(dst_dir, no_meta=False, save_caption=False))

    return dst_dir


def classify_characters(args, src_dir, is_start_stage):
    """Classifies characters in the given source directory.

    Args:
        args: A Namespace object containing the command-line arguments.
        src_dir: The path to the source directory containing images to be classified.
        is_start_stage: Whether this is the start stage of the pipeline.

    Returns:
        The path to the directory containing the classified images.
    """

    dst_dir = os.path.join(args.dst_dir, "intermediate", args.image_type, "classified")
    os.makedirs(dst_dir, exist_ok=True)

    # Determine whether to move or copy files to the destination directory.
    if src_dir == dst_dir:
        move = True
    else:
        move = args.remove_intermediate

    # Log information about the classification process.
    logging.info(f"Classifying characters to {dst_dir} ...")

    # Call the `classify_from_directory` function with the specified parameters.
    classify_from_directory(
        src_dir,
        dst_dir,
        ref_dir=args.character_ref_dir,
        ignore_character_metadata=args.ignore_character_metadata,
        to_extract_from_noise=not args.no_extract_from_noise,
        to_filter=not args.no_filter_characters,
        keep_unnamed=args.keep_unnamed,
        clu_min_samples=args.cluster_min_samples,
        merge_threshold=args.cluster_merge_threshold,
        same_threshold_rel=args.same_threshold_rel,
        same_threshold_abs=args.same_threshold_abs,
        move=move,
    )

    return os.path.dirname(dst_dir)


def select_images_for_dataset(args, src_dir, is_start_stage):
    classified_dir = os.path.join(src_dir, "classified")
    full_dir = os.path.join(src_dir, "raw")
    dst_dir = os.path.join(args.dst_dir, "training", args.image_type)
    os.makedirs(dst_dir, exist_ok=True)

    logging.info(f"Preparing dataset images to {dst_dir} ...")

    if is_start_stage:
        # rearrange json and ccip in case of manual inspection
        rearrange_related_files(classified_dir)

    overwrite_uncropped = (
        args.pipeline_type == "screenshots" or args.character_overwrite_uncropped
    )
    # update metadata using folder name
    characters = save_characters_to_meta(classified_dir, overwrite_uncropped)

    # save trigger word info
    trigger_word_filepath = os.path.join(dst_dir, "emb_init.csv")
    update_trigger_word_info(
        trigger_word_filepath,
        characters,
        args.image_type,
        args.overwrite_trigger_word_info,
    )

    if args.use_3stage_crop:
        logging.info(f"Performing 3 stage cropping for {classified_dir} ...")
        overwrite_path = is_start_stage and args.overwrite_path
        source = LocalSource(classified_dir, overwrite_path=overwrite_path)
        source.attach(
            ThreeStageSplitAction(split_person=False),
        ).export(SaveExporter(classified_dir, in_place=True))

    n_reg = args.n_anime_reg if args.pipeline_type == "screenshots" else 0
    # select images, resize, and save to training
    resize_character_images(
        [classified_dir, full_dir],
        dst_dir,
        max_size=args.max_size,
        ext=args.image_save_ext,
        image_type=args.image_type,
        n_nocharacter_frames=n_reg,
        to_resize=not args.no_resize,
    )

    if args.filter_again:
        logging.info(f"Removing duplicates from {dst_dir} ...")
        model = foz.load_zoo_model(args.detect_duplicate_model)
        for folder in ["cropped", "full", "no_characters"]:
            remove_similar_from_dir(
                os.path.join(dst_dir, folder), model=model, thresh=args.similar_thresh
            )
    if args.remove_intermediate:
        shutil.rmtree(classified_dir)
    return dst_dir


def tag_and_caption(args, src_dir, is_start_stage):
    if is_start_stage:
        # rearrange json and ccip in case of manual inspection
        rearrange_related_files(src_dir)

    if "character" in args.pruned_mode:
        if args.drop_hard_character_tags:
            drop_difficulty = 2
        else:
            drop_difficulty = 1
        # TODO: Deal with emb init difficulty later
        char_tag_proc = CharacterTagProcessor(drop_difficulty, emb_init_difficutly=0)
    else:
        char_tag_proc = None
    dst_dir = os.path.join(args.dst_dir, "training", args.image_type)
    core_tag_path = os.path.join(dst_dir, "core_tags.json")
    wildcard_path = os.path.join(dst_dir, "wildcard.txt")

    if args.process_from_original_tags or args.overwrite_tags:
        tags_attribute = "tags"
    else:
        tags_attribute = "processed_tags"
    if args.pruned_mode == "character_core":
        pruned_mode = "minimal"
    else:
        pruned_mode = args.pruned_mode

    with open(args.blacklist_tags_file, "r") as f:
        blacklisted_tags = {line.strip() for line in f}
    overlap_tags_dict = parse_overlap_tags(args.overlap_tags_file)
    overwrite_path = is_start_stage and args.overwrite_path

    source = LocalSource(src_dir, load_aux=args.load_aux, overwrite_path=overwrite_path)
    source = source.attach(
        TaggingAction(
            force=args.overwrite_tags,
            method=args.tagging_method,
            general_threshold=args.tag_threshold,
            character_threshold=1.01,
        ),
        TagPruningAction(
            blacklisted_tags,
            overlap_tags_dict,
            pruned_mode=pruned_mode,
            tags_attribute=tags_attribute,
            character_tag_processor=char_tag_proc,
        ),
        TagRemovingUnderscoreAction(),
    )
    logging.info(f"Tagging and captioning images in {src_dir} ...")

    if args.pruned_mode == "character_core":
        source.export(
            SaveExporter(src_dir, no_meta=False, save_caption=False, in_place=True)
        )
        if args.use_existing_core_tag_file:
            with open(core_tag_path, "r") as f:
                character_core_tags = json.load(f)
        else:
            assert char_tag_proc is not None
            character_core_tags = get_character_core_tags(
                src_dir, frequency_threshold=args.core_frequency_thresh
            )
            character_core_tags = char_tag_proc.categorize_character_tag_dict(
                character_core_tags
            )
            save_core_tag_info(character_core_tags, core_tag_path, wildcard_path)
        source = source.attach(
            CoreCharacterTagPruningAction(
                character_core_tags, tags_attribute="processed_tags"
            )
        )

    source = source.attach(
        TagSortingAction(args.sort_mode, max_tag_number=args.max_tag_number),
        CaptioningAction(args),
    )
    source.export(
        SaveExporter(
            src_dir,
            no_meta=False,
            save_caption=True,
            save_aux=args.save_aux,
            in_place=True,
        )
    )

    if args.pruned_mode != "character_core":
        get_character_core_tags_and_save(
            src_dir,
            core_tag_path,
            wildcard_path,
            frequency_threshold=args.core_frequency_thresh,
        )
    return src_dir


def rearrange(args, src_dir, is_start_stage):
    logging.info(f"Rearranging {src_dir} ...")
    if is_start_stage and args.load_aux:
        logging.info("Load metadata from auxiliary data ...")
        source = LocalSource(
            src_dir, load_aux=args.load_aux, overwrite_path=args.overwrite_path
        )
        source.export(
            SaveExporter(
                src_dir,
                no_meta=False,
                save_caption=True,
                save_aux=args.save_aux,
                in_place=True,
            )
        )
        rearrange_related_files(src_dir)
    arrange_folder(
        src_dir,
        src_dir,
        args.arrange_format,
        args.max_character_number,
        args.min_images_per_combination,
    )
    return os.path.join(args.dst_dir, "training")


def balance(args, src_dir, is_start_stage):
    training_dir = src_dir
    logging.info(f"Computing repeat for {training_dir} ...")
    if is_start_stage and args.load_aux:
        logging.info("Load metadata from auxiliary data ...")
        source = LocalSource(
            src_dir, load_aux=args.load_aux, overwrite_path=args.overwrite_path
        )
        source.export(
            SaveExporter(
                src_dir,
                no_meta=False,
                save_caption=True,
                save_aux=args.save_aux,
                in_place=True,
            )
        )
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
    get_repeat(
        training_dir, weight_mapping, args.min_multiply, args.max_multiply, log_file
    )
    return training_dir


# Mapping stage numbers to their respective function names
STAGE_FUNCTIONS = {
    1: extract_frames,
    2: crop_characters,
    3: classify_characters,
    4: select_images_for_dataset,
    5: tag_and_caption,
    6: rearrange,
    7: balance,
}

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument(
        "--src_dir", default=".", help="Directory containing source files"
    )
    parser.add_argument("--dst_dir", default=".", help="Directory to save output files")
    parser.add_argument(
        "--start_stage", default="1", help="Stage or alias to start from"
    )
    parser.add_argument("--end_stage", default="4", help="Stage or alias to end at")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help=("Directory to save logs. " "Set to None or none to disable."),
    )
    parser.add_argument(
        "--log_prefix", type=str, default="logfile", help="Prefix for log files"
    )
    parser.add_argument(
        "--overwrite_path",
        action="store_true",
        help=(
            "Overwrite path in metadata if LocalSource is used "
            "in the first stage. Should never be used in general."
        ),
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default="screenshots",
        help="Image type that we are dealing with, used for folder name",
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default="screenshots",
        choices=["screenshots", "fanart"],
        help=(
            "Pipeline type that is used to construct dataset ",
            "Options are 'screenshots' and 'fanart'",
        ),
    )
    parser.add_argument(
        "--remove_intermediate",
        action="store_true",
        help="Whether to remove intermediate result or not "
        + "(results after stage 1 are always saved)",
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
        default=0.98,
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
        help="Whether to ignore existing character metadata during classification",
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
        "--keep_unnamed",
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
        default=10,
        help=(
            "The absolute threshold for determining whether images belong to "
            "the same cluster for noise extraction and filtering"
        ),
    )

    # Arguments for dataset construction
    parser.add_argument(
        "--overwrite_trigger_word_info",
        action="store_true",
        help="Overwrite existing trigger word csv",
    )
    parser.add_argument(
        "--character_overwrite_uncropped",
        action="store_true",
        help=(
            "Overwrite existing character metadata for uncropped images "
            "(only meaning ful for 'fanart' pipeline as this is always the case "
            "for 'screenshots' pipeline)"
        ),
    )
    parser.add_argument(
        "--no_resize", action="store_true", help="Do not perform image resizing"
    )
    parser.add_argument(
        "--filter_again", action="store_true", help="Filter repeated images again here"
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
        "--use_3stage_crop",
        action="store_true",
        help=(
            "Use 3 stage crop for halfbody and head crops. "
            "This is slow and should only be called once for a set of images."
        ),
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
        help="Method used for tagging.",
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
        help=("Mode to sort the tags. " "Options are 'score', 'shuffle', 'original'."),
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
        default="tag_filtering/blacklist_tags.txt",
        help="Path to the file containing blacklisted tags.",
    )
    parser.add_argument(
        "--overlap_tags_file",
        type=str,
        default="tag_filtering/overlap_tags.json",
        help="Path to the file containing overlap tag information.",
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
            "Options are 'character', 'character_core', ''minimal', 'none'."
        ),
    )

    # Specific arguments for core tag processing
    parser.add_argument(
        "--core_frequency_thresh",
        type=float,
        default=0.5,
        help="Minimum frequency for a tag to be considered core tag.",
    )
    parser.add_argument(
        "--use_existing_core_tag_file",
        action="store_true",
        help=("Use existing core tag json instead of recomputing them."),
    )
    parser.add_argument(
        "--drop_hard_character_tags",
        action="store_true",
        help=(
            "Experimental. Whether to drop 'more difficult' character "
            "core tags or not. Ear, horn, and halo related tags are "
            "considered difficult for the moment."
        ),
    )

    # Arguments for captioning
    parser.add_argument(
        "--separator",
        type=str,
        default=",",
        help="Character used to separate items in captions",
    )
    parser.add_argument(
        "--caption_no_underscore",
        action="store_true",
        help="Do not include any underscore in captions",
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

    # For balancing
    parser.add_argument(
        "--min_multiply", type=float, default=1, help="Minimum multiply of each image"
    )
    parser.add_argument(
        "--max_multiply", type=int, default=100, help="Maximum multiply of each image"
    )
    parser.add_argument(
        "--weight_csv",
        default="csv_examples/default_weighting.csv",
        help="If provided use the provided csv to modify weights",
    )

    args = parser.parse_args()

    start_stage = args.start_stage
    end_stage = args.end_stage

    # Convert stage aliases to numbers if provided
    for stage_number in STAGE_ALIASES:
        if args.start_stage in STAGE_ALIASES[stage_number]:
            start_stage = stage_number
        if args.end_stage in STAGE_ALIASES[stage_number]:
            end_stage = stage_number

    start_stage = int(start_stage)
    end_stage = int(end_stage)

    src_dir = args.src_dir

    setup_logging(args.log_dir, args.log_prefix)

    # Loop through the stages and execute them
    for stage_num in range(start_stage, end_stage + 1):
        logging.info(f"-------------Start stage {stage_num}-------------")
        is_start_stage = stage_num == start_stage
        src_dir = STAGE_FUNCTIONS[stage_num](args, src_dir, is_start_stage)
