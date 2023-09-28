import os
import shutil
import logging
import argparse

import fiftyone.zoo as foz

from waifuc.action import PersonSplitAction, FaceCountAction, HeadCountAction
from waifuc.action import MinSizeFilterAction, NoMonochromeAction
from waifuc.action import TaggingAction

from anime2sd import extract_and_remove_similar, remove_similar_from_dir
from anime2sd import cluster_from_directory, classify_from_directory
from anime2sd import rearrange_related_files, save_characters_to_meta
from anime2sd import resize_character_images
from anime2sd import parse_overlap_tags

from anime2sd.waifuc_customize import LocalSource, SaveExporter
from anime2sd.waifuc_customize import TagPruningAction, TagSortingAction
from anime2sd.waifuc_customize import TagRemovingUnderscoreAction
from anime2sd.waifuc_customize import CaptioningAction


def extract_frames(args, src_dir):
    dst_dir = os.path.join(
        args.dst_dir, 'intermediate', args.image_type, 'raw')
    os.makedirs(dst_dir, exist_ok=True)
    logging.info(f'Extracting frames to {dst_dir} ...')
    extract_and_remove_similar(src_dir, dst_dir, args.prefix,
                               ep_init=args.ep_init,
                               model_name=args.detect_duplicate_model,
                               thresh=args.similar_thresh,
                               to_remove_similar=not args.no_remove_similar)


def crop_characters(args, src_dir):

    source = LocalSource(src_dir)
    source = source.attach(
        NoMonochromeAction(),
        PersonSplitAction(keep_original=False, level='n'),
        FaceCountAction(1, level='n'),
        HeadCountAction(1, level='n'),
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


def classify_characters(args, src_dir):
    # TODO: multi-stage classification
    # TODO: add cluster filter mechanism (min image size, max k)

    dst_dir = os.path.join(
        args.dst_dir, 'intermediate', args.image_type, 'classified')
    os.makedirs(dst_dir, exist_ok=True)
    if src_dir == dst_dir:
        move = True
    else:
        move = not args.save_intermediate

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


def select_images_for_dataset(args, src_dir):
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
    if not args.save_intermediate:
        shutil.rmtree(classified_dir)
        shutil.rmtree(full_dir)


def tag_and_caption(args, src_dir):
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

    logging.info('Tagging and captioning ...')
    source.export(SaveExporter(
        src_dir, no_meta=False,
        save_caption=True, save_aux=args.save_aux, in_place=True))


def rearrange_and_balance(args):
    pass


# Mapping stage numbers to their respective function names
STAGE_FUNCTIONS = {
    1: extract_frames,
    2: crop_characters,
    3: classify_characters,
    4: select_images_for_dataset,
    5: tag_and_caption,
    6: rearrange_and_balance,
}


# Mapping stage numbers to their aliases
STAGE_ALIASES = {
    1: ['extract'],
    2: ['crop'],
    3: ['classify'],
    4: ['select'],
    5: ['tag', 'caption', 'tag_and_caption'],
    6: ['arrange'],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--src_dir", default='.',
                        help="directory containing source files")
    parser.add_argument("--dst_dir", default='.',
                        help="directory to save output files")
    parser.add_argument("--start_stage", default="1",
                        help="Stage or alias to start from")
    parser.add_argument("--end_stage", default="4",
                        help="Stage or alias to end at")
    parser.add_argument(
        "--image_type", default="screenshots",
        help="Image type that we are dealing with, used for folder name")
    parser.add_argument(
        "--save_intermediate", action="store_true",
        help="Whether to save intermediate result or not "
        + "(results after stage 1 are always saved)")

    # Arguments for video extraction
    parser.add_argument("--prefix", default='', help="output file prefix")
    parser.add_argument("--ep_init", type=int, default=1,
                        help="episode number to start with")

    # Arguments for duplicate detection
    parser.add_argument("--no_remove_similar", action="store_true",
                        help="do not remove similar images")
    parser.add_argument("--detect_duplicate_model",
                        default='mobilenet-v2-imagenet-torch',
                        help="model used for duplicate detection")
    parser.add_argument(
        "--similar_thresh", type=float, default=0.985,
        help="cosine similarity threshold for image duplicate detection")

    # Arguments for character cropping
    parser.add_argument("--min_crop_size", type=int, default=320,
                        help="minimum size for character cropping")

    # Arguments for character clustering/classification
    parser.add_argument(
        "--cluster_merge_threshold", type=float, default=0.85,
        help="cluster merge threshold in character clusterining")
    parser.add_argument(
        "--cluster_min_samples", type=int, default=5,
        help="minimum cluster samples in character clusterining")
    # Important
    parser.add_argument(
        "--character_ref_dir", default=None,
        help="directory conaining reference character images")

    # Arguments for dataset construction
    parser.add_argument("--no_resize", action="store_true",
                        help="do not perform image resizing")
    parser.add_argument("--filter_again", action="store_true",
                        help="use lpips to filter repeated images here")
    parser.add_argument("--max_size", type=int, default=768,
                        help="max image size that shorter edge aligns to")
    parser.add_argument("--image_save_ext", default='.webp',
                        help="dataset image extensino")
    parser.add_argument("--n_anime_reg", type=int, default=500,
                        help="number of images with no characters to keep")

    # Loading and saving of metadata for tagging and captioning stage
    parser.add_argument('--load_aux', type=str, nargs='*',
                        default=['processed_tags', 'characters'],
                        help="List of auxiliary attributes to load. "
                        "Default is processed_tags and characters. "
                        "E.g., --load_aux attr1 attr2 attr3")
    parser.add_argument('--save_aux', type=str, nargs='*',
                        default=['processed_tags', 'characters'],
                        help="List of auxiliary attributes to save. "
                        "Default is processed_tags and characters. "
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
        '--max_tag_number', type=int, default=15,
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

    args = parser.parse_args()

    start_stage = args.start_stage
    end_stage = args.end_stage

    # Convert stage aliases to numbers if provided
    for stage_number in STAGE_ALIASES:
        if args.start_stage in STAGE_ALIASES[stage_number]:
            start_stage = stage_number
        if args.end_stage in STAGE_ALIASES[stage_number]:
            end_stage = stage_number

    src_dir = args.src_dir

    logging.getLogger().setLevel(logging.INFO)

    # Loop through the stages and execute them
    for stage_num in range(int(start_stage), int(end_stage) + 1):
        logging.info(f'-------------Start stage {stage_num}-------------')
        src_dir = STAGE_FUNCTIONS[stage_num](args, src_dir)
