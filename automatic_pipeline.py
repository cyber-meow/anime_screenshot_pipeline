import os
import logging
import argparse

from waifuc.source import LocalSource
from waifuc.export import SaveExporter
from waifuc.action import PersonSplitAction, FaceCountAction, HeadCountAction
from waifuc.action import MinSizeFilterAction, NoMonochromeAction
from waifuc.action import FilterSimilarAction

from anime2sd import extract_and_remove_similar
from anime2sd import cluster_from_directory, classify_from_directory


def extract_frames(args, src_dir):
    dst_dir = os.path.join(args.dst_dir, 'intermediate', 'raw')
    os.makedirs(dst_dir, exist_ok=True)
    logging.info(f'Extracting frames to {dst_dir} ...')
    extract_and_remove_similar(src_dir, dst_dir, args.prefix,
                               args.ep_init, thresh=args.similar_thresh,
                               to_remove_similar=not args.no_remove_similar)


def crop_characters(args, src_dir):

    source = LocalSource(src_dir)
    source = source.attach(
        NoMonochromeAction(),
        PersonSplitAction(keep_original=False, level='n'),
        FaceCountAction(1, level='n'),
        HeadCountAction(1, level='n'),
        MinSizeFilterAction(args.crop_min_size),
        FilterSimilarAction('all'),
    )

    dst_dir = os.path.join(os.path.dirname(src_dir), 'cropped')
    os.makedirs(dst_dir, exist_ok=True)
    logging.info(f'Cropping individual characters to {dst_dir} ...')
    source.export(SaveExporter(src_dir, no_meta=False))

    return dst_dir


def classify_characters(args, src_dir):

    dst_dir = os.path.join(os.path.dirname(src_dir), 'classified')
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


def construct_dataset(args):
    pass


def tag_images(args):
    pass


def generate_captions(tags):
    pass


def rearrange_and_balance(args):
    pass


# Mapping stage numbers to their respective function names
STAGE_FUNCTIONS = {
    1: extract_frames,
    2: crop_characters,
    3: classify_characters,
    4: construct_dataset,
    5: tag_images,
    6: generate_captions,
    7: rearrange_and_balance,
}


# Mapping stage numbers to their aliases
STAGE_ALIASES = {
    1: "extract",
    2: "process",
    3: "analyze"
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
    parser.add_argument("--end_stage", default="1",
                        help="Stage or alias to end at")
    parser.add_argument(
        "--save_intermediate", default="1",
        help="Whether to save intermediate result or not "
        + "(results after stage 1 are always saved)")

    # Arguments for video extraction
    parser.add_argument("--prefix", default='', help="output file prefix")
    parser.add_argument("--ep_init", type=int, default=1,
                        help="episode number to start with")

    # Arguments for duplicate detection
    parser.add_argument("--no-remove-similar", action="store_true",
                        help="flag to not remove similar images")
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

    # Loop through the stages and execute them
    for stage_num in range(int(start_stage), int(end_stage) + 1):
        src_dir = STAGE_FUNCTIONS[stage_num](args, src_dir)
