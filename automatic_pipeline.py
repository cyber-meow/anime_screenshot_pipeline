import os
import shutil
import logging
import argparse
from datetime import datetime

import fiftyone.zoo as foz

from waifuc.action import PersonSplitAction
# from waifuc.action import FaceCountAction, HeadCountAction
from waifuc.action import MinSizeFilterAction, NoMonochromeAction
from waifuc.action import TaggingAction

from anime2sd import extract_and_remove_similar, remove_similar_from_dir
from anime2sd import cluster_from_directory, classify_from_directory
from anime2sd import rearrange_related_files, save_characters_to_meta
from anime2sd import resize_character_images
from anime2sd import parse_overlap_tags, read_weight_mapping
from anime2sd import arrange_folder, get_repeat

from anime2sd.waifuc_customize import LocalSource, SaveExporter
from anime2sd.waifuc_customize import TagPruningAction, TagSortingAction
from anime2sd.waifuc_customize import TagRemovingUnderscoreAction
from anime2sd.waifuc_customize import CaptioningAction


def setup_logging(log_dir, log_prefix):
    """
    Set up logging to file and stdout with specified directory and prefix.

    :param log_dir: Directory to save the log file.
    :param log_prefix: Prefix for the log file name.
    :return: None
    """

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Add formatter to ch
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Create file handler and set level to info
    if log_dir not in ['None', 'none']:
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now()
        str_current_time = str(current_time)
        log_file = os.path.join(
            log_dir, f"{log_prefix}_{str_current_time}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        fh.setFormatter(formatter)


def extract_frames(args, src_dir, is_start_stage):
    dst_dir = os.path.join(
        args.dst_dir, 'intermediate', args.image_type, 'raw')
    os.makedirs(dst_dir, exist_ok=True)
    logging.info(f'Extracting frames to {dst_dir} ...')
    extract_and_remove_similar(src_dir, dst_dir, args.image_prefix,
                               ep_init=args.ep_init,
                               model_name=args.detect_duplicate_model,
                               thresh=args.similar_thresh,
                               to_remove_similar=not args.no_remove_similar)


def crop_characters(args, src_dir, is_start_stage):

    source = LocalSource(src_dir)
    source = source.attach(
        NoMonochromeAction(),
        PersonSplitAction(keep_original=False, level='n'),
        # TODO: investigate the problem
        # This seems to filter out some character with more special appearance
        # FaceCountAction(1, level='n'),
        # HeadCountAction(1, level='n'),
        MinSizeFilterAction(args.min_crop_size),
        # Not used here because it can be problematic for multi-character scene
        # Some not moving while other moving
        # FilterSimilarAction('all'),
    )

    dst_dir = os.path.join(
        args.dst_dir, 'intermediate', args.image_type, 'cropped')
    os.makedirs(dst_dir, exist_ok=True)
    logging.info(f'Cropping individual characters to {dst_dir} ...')
    source.export(
        SaveExporter(dst_dir, no_meta=False, save_caption=False))

    return dst_dir


def classify_characters(args, src_dir, is_start_stage):
    # TODO: multi-stage classification
    # TODO: add cluster filter mechanism (min image size, max k)

    dst_dir = os.path.join(
        args.dst_dir, 'intermediate', args.image_type, 'classified')
    os.makedirs(dst_dir, exist_ok=True)
    if src_dir == dst_dir:
        move = True
    else:
        move = args.remove_intermediate

    if args.character_ref_dir is None:
        logging.info(f'Clustering characters to {dst_dir} ...')
        cluster_from_directory(
            src_dir, dst_dir,
            args.cluster_merge_threshold,
            clu_min_samples=args.cluster_min_samples,
            move=move)

    else:
        classify_from_directory(
            src_dir, dst_dir,
            args.character_ref_dir,
            clu_min_samples=args.cluster_min_samples,
            move=move)
    return os.path.dirname(dst_dir)


def select_images_for_dataset(args, src_dir, is_start_stage):

    classified_dir = os.path.join(src_dir, 'classified')
    full_dir = os.path.join(src_dir, 'raw')
    dst_dir = os.path.join(args.dst_dir, 'training', args.image_type)

    logging.info(f'Preparing dataset images to {dst_dir} ...')
    # rearrange json and ccip in case of manual inspection
    rearrange_related_files(classified_dir)
    # update metadata using folder name
    save_characters_to_meta(classified_dir)
    # select images, resize, and save to training
    resize_character_images(classified_dir, full_dir, dst_dir,
                            max_size=args.max_size,
                            ext=args.image_save_ext,
                            image_type=args.image_type,
                            n_nocharacter_frames=args.n_anime_reg,
                            to_resize=not args.no_resize)

    if args.filter_again:
        logging.info(f'Removing duplicates from {dst_dir} ...')
        model = foz.load_zoo_model(args.detect_duplicate_model)
        for folder in ['cropped', 'full', 'no_characters']:
            remove_similar_from_dir(os.path.join(dst_dir, folder),
                                    model=model,
                                    thresh=args.similar_thresh)
    if args.remove_intermediate:
        shutil.rmtree(classified_dir)
    return dst_dir


def tag_and_caption(args, src_dir, is_start_stage):

    with open(args.blacklist_tags_file, 'r') as f:
        blacklisted_tags = {line.strip() for line in f}
    overlap_tags_dict = parse_overlap_tags(args.overlap_tags_file)
    if args.process_from_original_tags or args.overwrite_tags:
        tags_attribute = 'tags'
    else:
        tags_attribute = 'processed_tags'
    source = LocalSource(src_dir, load_aux=args.load_aux)
    source = source.attach(
        TaggingAction(
            force=args.overwrite_tags,
            method=args.tagging_method,
            general_threshold=args.tag_threshold,
            character_threshold=1.01),
        TagPruningAction(
            blacklisted_tags,
            overlap_tags_dict,
            pruned_type=args.pruned_type,
            tags_attribute=tags_attribute),
        TagSortingAction(
            args.sort_mode,
            max_tag_number=args.max_tag_number),
        TagRemovingUnderscoreAction(),
        CaptioningAction(args),
    )

    logging.info(f'Tagging and captioning images in {src_dir} ...')
    source.export(SaveExporter(
        src_dir, no_meta=False,
        save_caption=True, save_aux=args.save_aux, in_place=True))
    return src_dir


def rearrange(args, src_dir, is_start_stage):
    logging.info(f'Rearranging {src_dir} ...')
    if is_start_stage:
        logging.info('Load metadata from auxiliary data ...')
        source = LocalSource(src_dir, load_aux=args.load_aux)
        source.export(SaveExporter(
            src_dir, no_meta=False,
            save_caption=True,
            save_aux=args.save_aux, in_place=True))
    arrange_folder(
        src_dir, src_dir, args.arrange_format,
        args.max_character_number, args.min_images_per_combination)
    return src_dir


def balance(args, src_dir, is_start_stage):
    training_dir = os.path.join(args.dst_dir, 'training')
    logging.info(f'Computing repeat for {training_dir} ...')
    if is_start_stage:
        logging.info('Load metadata from auxiliary data ...')
        source = LocalSource(src_dir, load_aux=args.load_aux)
        source.export(SaveExporter(
            src_dir, no_meta=False,
            save_caption=True,
            save_aux=args.save_aux, in_place=True))
    if args.weight_csv is not None:
        weight_mapping = read_weight_mapping(args.weight_csv)
    else:
        weight_mapping = None
    current_time = datetime.now()
    str_current_time = str(current_time)
    log_file = os.path.join(
        args.log_dir, f"{args.log_prefix}_weighting_{str_current_time}.log")
    get_repeat(
        training_dir, weight_mapping,
        args.min_multiply, args.max_multiply, log_file)
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
    1: ['extract'],
    2: ['crop'],
    3: ['classify'],
    4: ['select'],
    5: ['tag', 'caption', 'tag_and_caption'],
    6: ['arrange'],
    7: ['balance'],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--src_dir", default='.',
                        help="Directory containing source files")
    parser.add_argument("--dst_dir", default='.',
                        help="Directory to save output files")
    parser.add_argument("--start_stage", default="1",
                        help="Stage or alias to start from")
    parser.add_argument("--end_stage", default="4",
                        help="Stage or alias to end at")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--log_prefix', type=str, default='logfile',
                        help='Prefix for log files')
    parser.add_argument(
        "--image_type", default="screenshots",
        help="Image type that we are dealing with, used for folder name")
    parser.add_argument(
        "--remove_intermediate", action="store_true",
        help="Whether to remove intermediate result or not "
        + "(results after stage 1 are always saved)")

    # Arguments for video extraction
    parser.add_argument("--image_prefix", default='',
                        help="Output image prefix")
    parser.add_argument("--ep_init", type=int, default=1,
                        help="Episode number to start with")

    # Arguments for duplicate detection
    parser.add_argument("--no_remove_similar", action="store_true",
                        help="Do not remove similar images")
    parser.add_argument("--detect_duplicate_model",
                        default='mobilenet-v2-imagenet-torch',
                        help="Model used for duplicate detection")
    parser.add_argument(
        "--similar_thresh", type=float, default=0.985,
        help="Cosine similarity threshold for image duplicate detection")

    # Arguments for character cropping
    parser.add_argument("--min_crop_size", type=int, default=320,
                        help="Minimum size for character cropping")

    # Arguments for character clustering/classification
    parser.add_argument(
        "--cluster_merge_threshold", type=float, default=0.85,
        help="Cluster merge threshold in character clusterining")
    parser.add_argument(
        "--cluster_min_samples", type=int, default=5,
        help="Minimum cluster samples in character clusterining")
    # Important
    parser.add_argument(
        "--character_ref_dir", default=None,
        help="Directory conaining reference character images")

    # Arguments for dataset construction
    parser.add_argument("--no_resize", action="store_true",
                        help="Do not perform image resizing")
    parser.add_argument("--filter_again", action="store_true",
                        help="Filter repeated images again here")
    parser.add_argument("--max_size", type=int, default=768,
                        help="Max image size that shorter edge aligns to")
    parser.add_argument("--image_save_ext", default='.webp',
                        help="Dataset image extensino")
    parser.add_argument("--n_anime_reg", type=int, default=500,
                        help="Number of images with no characters to keep")

    # Loading and saving of metadata for tagging and captioning stage
    parser.add_argument('--load_aux', type=str, nargs='*',
                        # default=['processed_tags', 'characters'],
                        help="List of auxiliary attributes to load. "
                        # "Default is processed_tags and characters. "
                        "E.g., --load_aux attr1 attr2 attr3")
    parser.add_argument('--save_aux', type=str, nargs='*',
                        # default=['processed_tags', 'characters'],
                        help="List of auxiliary attributes to save. "
                        # "Default is processed_tags and characters. "
                        "E.g., --save_aux attr1 attr2 attr3")

    # Arguments for tagging
    parser.add_argument('--overwrite_tags', action="store_true",
                        help="Whether to overwrite existing tags.")
    parser.add_argument('--tagging_method', type=str,
                        default='wd14_convnextv2',
                        help="Method used for tagging.")
    parser.add_argument('--tag_threshold', type=float, default=0.35,
                        help="Threshold for tagging.")

    # Arguments for tag processing
    parser.add_argument(
        '--process_from_original_tags', action="store_true",
        help="process tags from original tags instead of processed tags"
    )
    parser.add_argument(
        '--sort_mode', type=str, default='score',
        choices=['score', 'shuffle', 'original'],
        help=("Mode to sort the tags. "
              "Options are 'score', 'shuffle', 'original'.")
    )
    parser.add_argument(
        '--pruned_type', type=str, default='character',
        choices=['character', 'minimal', 'none'],
        help=("Type of tags to be pruned. "
              "Options are 'character', 'minimal', 'none'.")
    )
    parser.add_argument(
        '--max_tag_number', type=int, default=30,
        help="Max number of tags to include in caption."
    )
    parser.add_argument(
        '--blacklist_tags_file', type=str,
        default='tag_filtering/blacklist.txt',
        help="Path to the file containing blacklisted tags."
    )
    parser.add_argument(
        '--overlap_tags_file', type=str,
        default='tag_filtering/overlap_tags.json',
        help="Path to the file containing overlap tag information."
    )

    # Arguments for captioning
    parser.add_argument(
        "--separator", type=str, default=',',
        help="Character used to separate items in captions")
    parser.add_argument(
        "--caption_no_underscore", action="store_true",
        help="Do not include any underscore in captions")
    parser.add_argument(
        "--use_npeople_prob", type=float, default=0,
        help="Probability to include number of people in captions")
    parser.add_argument(
        "--use_character_prob", type=float, default=1,
        help="Probability to include character info in captions")
    parser.add_argument(
        "--use_copyright_prob", type=float, default=0,
        help="Probability to include copyright info in captions")
    parser.add_argument(
        "--use_image_type_prob", type=float, default=1,
        help="Probability to include image type info in captions")
    parser.add_argument(
        "--use_artist_prob", type=float, default=0,
        help="Probability to include artist info in captions")
    parser.add_argument(
        "--use_rating_prob", type=float, default=1,
        help="Probability to include rating info in captions")
    parser.add_argument(
        "--use_tags_prob", type=float, default=1,
        help="Probability to include tag info in captions")

    # Arguments for folder organization
    parser.add_argument(
        "--arrange_format", type=str, default='n_characters/character',
        help='Description of the concept balancing directory hierarchy'
    )
    parser.add_argument(
        "--max_character_number", type=int, default=6,
        help="If have more than X characters put X+")
    parser.add_argument(
        "--min_images_per_combination", type=int, default=10,
        help=("Put others instead of character name if number of images "
              "of the character combination is smaller then this number"))

    # For balancing
    parser.add_argument(
        '--min_multiply', type=float, default=1,
        help='Minimum multiply of each image')
    parser.add_argument(
        '--max_multiply', type=int, default=100,
        help='Maximum multiply of each image')
    parser.add_argument(
        '--weight_csv', default='csv_examples/default_weighting.csv',
        help='If provided use the provided csv to modify weights')

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
        logging.info(f'-------------Start stage {stage_num}-------------')
        is_start_stage = stage_num == start_stage
        src_dir = STAGE_FUNCTIONS[stage_num](args, src_dir, is_start_stage)
