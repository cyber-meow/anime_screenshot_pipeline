import os
import toml
import copy
import shutil
import logging
import argparse
from datetime import datetime

import asyncio
import concurrent.futures

from waifuc.action import PersonSplitAction
from waifuc.action import MinSizeFilterAction
from waifuc.action import ThreeStageSplitAction

from anime2sd import download_animes, download_images
from anime2sd import extract_and_remove_similar
from anime2sd import classify_from_directory
from anime2sd import select_dataset_images_from_directory
from anime2sd import (
    tag_and_caption_from_directory,
    compute_and_save_core_tags,
    tag_and_caption_from_directory_core_final,
)
from anime2sd import arrange_folder
from anime2sd import read_weight_mapping, get_repeat
from anime2sd import DuplicateRemover
from anime2sd import CharacterTagProcessor, TaggingManager, CaptionGenerator

from anime2sd.basics import (
    read_class_mapping,
    rearrange_related_files,
    load_metadata_from_aux,
)
from anime2sd.execution_ordering import (
    setup_logging,
    get_src_dir,
    get_and_create_dst_dir,
    get_execution_configs,
)
from anime2sd.parse_arguments import parse_arguments
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
                    if not hasattr(new_args, nested_key):
                        logging.warning(f"Key {nested_key} is not a valid argument.")
                    setattr(new_args, nested_key, nested_value)
            else:
                if not hasattr(new_args, key):
                    logging.warning(f"Key {key} is not a valid argument.")
                setattr(new_args, key, value)
    except Exception as e:
        print(f"Error loading config from {toml_path}: {e}")
    return new_args


def setup_args(args):
    """
    Sets up the start and end stages for the pipeline based on the provided arguments.
    """
    # Mapping stage numbers to their aliases
    STAGE_ALIASES = {
        0: ["download"],
        1: ["extract", "remove_similar", "remove_duplicates"],
        2: ["crop"],
        3: ["classify"],
        4: ["select"],
        5: ["tag", "caption", "tag_and_caption"],
        6: ["arrange"],
        7: ["balance", "compute_multiply"],
    }
    if not args.image_type:
        args.image_type = config.pipeline_type
    if not config.anime_name_booru:
        args.anime_name_booru = args.anime_name
    if not config.log_prefix:
        if args.anime_name:
            args.log_prefix = args.anime_name
        else:
            args.log_prefix = "logfile"

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


def download(args, stage, logger):
    """
    Downloads images or animes from the internet.
    """
    if args.pipeline_type == "screenshots":
        dst_dir = get_and_create_dst_dir(args, "intermediate", "animes")
        if not args.anime_name:
            raise ValueError("Anime name must be provided for anime downloading")

        logger.info(f"Downloading animes to {dst_dir} ...")
        download_animes(
            dst_dir,
            anime_names=[args.anime_name],
            candidate_submitters=args.candidate_submitters,
            resolution=args.anime_resolution,
            min_download_episode=args.min_download_episode,
            max_download_episode=args.max_download_episode,
            logger=logger,
        )

    elif args.pipeline_type == "booru":
        dst_dir = get_and_create_dst_dir(args, "intermediate", "raw")

        if args.character_info_file is not None and os.path.exists(
            args.character_info_file
        ):
            character_mapping = read_class_mapping(args.character_info_file)
        else:
            character_mapping = None
        anime = [args.anime_name_booru] if args.anime_name_booru else []
        if not (anime or character_mapping):
            raise ValueError(
                "Either anime or character info should be provided for booru "
                " downloading (by specifying --anime_name, --anime_name_booru "
                "or --character_info_file)"
            )

        logger.info(f"Downloading images to {dst_dir} ...")
        download_images(
            dst_dir,
            anime,
            limit_all=args.booru_download_limit,
            limit_per_character=args.booru_download_limit_per_character,
            ratings=args.allowed_ratings,
            classes=args.allowed_image_classes,
            max_image_size=args.max_download_size,
            character_mapping=character_mapping,
            download_for_characters=args.download_for_characters,
            save_aux=args.save_aux,
            logger=logger,
        )


def extract_frames_and_or_remove_similar(args, stage, logger):
    """
    Extracts frames from videos and saves them to the destination directory.
    This function also handles duplicate detection and removal.
    """
    # Get the path to the source directory containing the videos
    src_dir = get_src_dir(args, stage)

    if args.no_remove_similar:
        duplicate_remover = None
    else:
        duplicate_remover = DuplicateRemover(
            args.detect_duplicate_model,
            threshold=args.similar_thresh,
            dataloader_batch_size=args.detect_duplicate_batch_size,
            logger=logger,
        )

    if args.pipeline_type == "screenshots":
        # Get the path to the destination directory for the extracted frames
        dst_dir = get_and_create_dst_dir(args, "intermediate", "raw")
        logger.info(f"Extracting frames to {dst_dir} ...")

        extract_and_remove_similar(
            src_dir,
            dst_dir,
            args.image_prefix,
            ep_init=args.ep_init,
            extract_key=args.extract_key,
            duplicate_remover=duplicate_remover,
            logger=logger,
        )
    else:
        duplicate_remover.remove_similar_from_dir(src_dir)


# TODO: Avoid cropping for already cropped data
def crop_characters(args, stage, logger):
    """Crops individual characters from images in the source directory."""
    # Get the path to the source directory containing the images to crop from
    src_dir = get_src_dir(args, stage)
    # Get the path to the destination directory for the cropped images
    dst_dir = get_and_create_dst_dir(args, "intermediate", "cropped")
    logger.info(f"Cropping individual characters to {dst_dir} ...")

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


def classify_characters(args, stage, logger):
    """Classifies characters in the given source directory."""

    # Get the path to the source directory containing images to be classified
    src_dir = get_src_dir(args, stage)
    # Get the path to the destination directory containing the classified images
    dst_dir = get_and_create_dst_dir(args, "intermediate", "classified")

    # Determine whether to move or copy files to the destination directory.
    move = args.remove_intermediate or (src_dir == dst_dir)
    # Determine whether to ignore existing character metadata.
    ignore_character_metadata = (
        args.ignore_character_metadata or args.pipeline_type == "screenshots"
    )

    # Log information about the classification process.
    logger.info(f"Classifying characters to {dst_dir} ...")

    # Call the `classify_from_directory` function with the specified parameters.
    classify_from_directory(
        src_dir,
        dst_dir,
        ref_dir=args.character_ref_dir,
        ignore_character_metadata=ignore_character_metadata,
        to_extract_from_noise=not args.no_extract_from_noise,
        to_filter=not args.no_filter_characters,
        keep_unnamed=args.keep_unnamed_clusters,
        accept_multiple_candidates=args.accept_multiple_candidates,
        clu_min_samples=args.cluster_min_samples,
        merge_threshold=args.cluster_merge_threshold,
        same_threshold_rel=args.same_threshold_rel,
        same_threshold_abs=args.same_threshold_abs,
        n_add_images_to_ref=args.n_add_to_ref_per_character,
        move=move,
        logger=logger,
    )


def select_dataset_images(args, stage, logger):
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
        rearrange_related_files(classified_dir, logger)
    overwrite_path = is_start_stage and args.overwrite_path

    if args.filter_again:
        duplicate_remover = DuplicateRemover(
            args.detect_duplicate_model,
            threshold=args.similar_thresh,
            dataloader_batch_size=args.detect_duplicate_batch_size,
            logger=logger,
        )
    else:
        duplicate_remover = None

    logger.info(f"Preparing dataset images to {dst_dir} ...")

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
        duplicate_remover=duplicate_remover,
        logger=logger,
    )

    if args.remove_intermediate:
        shutil.rmtree(classified_dir)


async def tag_and_caption(
    args, stage, config_index, execution_config, stage_events, executor, logger
):
    """
    Perform in-place tagging and captioning.
    Note that this is a coroutine because we may need to wait for other
    pipelines to perform the first stage of tagging before computing the core
    tags when prune_mode is 'character_core'.
    """
    # Get path to the directiry containing images to be tagged and captioned
    src_dir = get_src_dir(args, stage)
    is_start_stage = args.start_stage == stage
    if is_start_stage:
        # rearrange json and ccip in case of manual inspection
        rearrange_related_files(src_dir, logger)
    overwrite_path = is_start_stage and args.overwrite_path

    if "character" in args.prune_mode:
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
        prune_mode=args.prune_mode,
        blacklist_tags_file=args.blacklist_tags_file,
        overlap_tags_file=args.overlap_tags_file,
        character_tag_processor=char_tag_proc,
        process_from_original_tags=args.process_from_original_tags,
        sort_mode=args.sort_mode,
        max_tag_number=args.max_tag_number,
        logger=logger,
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

    logger.info(f"Tagging and captioning images in {src_dir} ...")

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(
        executor,
        tag_and_caption_from_directory,
        src_dir,
        tagging_manager,
        caption_generator,
        args.load_aux,
        args.save_aux,
        overwrite_path,
        logger,
    )
    stage_events[config_index]["5_phase1"].set()

    core_tag_dir = get_src_dir(args, "core_tag")
    core_tag_path = os.path.join(core_tag_dir, "core_tag.json")

    if execution_config.run_save_core:
        await asyncio.gather(
            *(
                stage_events[dep_index]["5_phase1"].wait()
                for dep_index in execution_config.save_core_dependencies
            )
        )
        await loop.run_in_executor(
            executor,
            compute_and_save_core_tags,
            core_tag_dir,
            core_tag_path,
            args.core_frequency_thresh,
            char_tag_proc,
            caption_generator,
            execution_config.image_types,
            args.overwrite_emb_init_info,
            logger,
        )
    stage_events[config_index]["save_core"].set()

    if args.prune_mode == "character_core":
        await asyncio.gather(
            *(
                stage_events[dep_index]["save_core"].wait()
                for dep_index in execution_config.stage5_final_dependencies
            )
        )
        await loop.run_in_executor(
            executor,
            tag_and_caption_from_directory_core_final,
            src_dir,
            core_tag_path,
            tagging_manager,
            caption_generator,
            args.load_aux,
            args.save_aux,
            logger,
        )


def rearrange(args, stage, logger):
    """Rearrange the images in the directory."""
    # Get path to the directiry containing images to be rearranged
    src_dir = get_src_dir(args, stage)
    logger.info(f"Rearranging {src_dir} ...")
    if args.start_stage == stage and args.load_aux:
        load_metadata_from_aux(
            src_dir, args.load_aux, args.save_aux, args.overwrite_path, logger=logger
        )
        rearrange_related_files(src_dir, logger)
    arrange_folder(
        src_dir,
        src_dir,
        args.arrange_format,
        args.max_character_number,
        args.min_images_per_combination,
        logger=logger,
    )


def balance(args, stage, logger):
    """Compute the repeat for the images in the directory."""
    # Get path to the directiry containing images for which repeat needs to be computed
    src_dir = get_src_dir(args, stage)
    if args.start_stage == stage and args.load_aux:
        load_metadata_from_aux(
            src_dir, args.load_aux, args.save_aux, args.overwrite_path, logger=logger
        )
        rearrange_related_files(src_dir, logger)
    logger.info(f"Computing repeat for {src_dir} ...")
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
        src_dir,
        weight_mapping,
        args.min_multiply,
        args.max_multiply,
        log_file,
        logger=logger,
    )


def run_stage(config, stage_num, logger):
    # Mapping stage numbers to their respective function names
    STAGE_FUNCTIONS = {
        0: download,
        1: extract_frames_and_or_remove_similar,
        2: crop_characters,
        3: classify_characters,
        4: select_dataset_images,
        # 5: tag_and_caption,
        6: rearrange,
        7: balance,
    }

    STAGE_FUNCTIONS[stage_num](config, stage_num, logger)


async def run_pipeline(config, config_index, execution_config, stage_events, executor):
    logger = setup_logging(
        config.log_dir,
        f"{config.pipeline_type}_{config.log_prefix}",
        f"pipeline_{config_index}",
    )
    # Loop through the stages and execute them
    for stage_num in range(config.start_stage, config.end_stage + 1):
        if stage_num == 3:
            # Wait for dependent booru configs to complete classification
            await asyncio.gather(
                *(
                    stage_events[dep_index][3].wait()
                    for dep_index in execution_config.stage3_dependencies
                )
            )
        if stage_num == 6:
            if not execution_config.run_stage6:
                stage_events[config_index][stage_num].set()
                continue
            # Wait for dependent configs to complete tagging
            await asyncio.gather(
                *(
                    stage_events[dep_index][5].wait()
                    for dep_index in execution_config.stage6_dependencies
                )
            )
        if stage_num == 7:
            if not execution_config.run_stage7:
                stage_events[config_index][stage_num].set()
                continue
            # Wait for dependent configs to complete rearranging
            await asyncio.gather(
                *(
                    stage_events[dep_index][6].wait()
                    for dep_index in execution_config.stage7_dependencies
                )
            )

        logger.info(f"-------------Start stage {stage_num}-------------")
        loop = asyncio.get_running_loop()
        if stage_num == 5:
            await tag_and_caption(
                config,
                stage_num,
                config_index,
                execution_config,
                stage_events,
                executor,
                logger,
            )
        else:
            await loop.run_in_executor(executor, run_stage, config, stage_num, logger)
        stage_events[config_index][stage_num].set()


async def main(configs):
    # Set up configs for execution dependencies and optional skips
    execution_configs = get_execution_configs(configs)

    # Initialize events for each stage of each config
    stage_events = []
    for config in configs:
        events = {
            j: asyncio.Event() for j in range(config.start_stage, config.end_stage + 1)
        }
        # Add events to manage core tag computation and saving for stage 5
        if 5 in events.keys():
            events["5_phase1"] = asyncio.Event()
            events["save_core"] = asyncio.Event()
        stage_events.append(events)

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Run pipelines asynchronously with dependencies
        await asyncio.gather(
            *(
                run_pipeline(
                    config, config_index, execution_config, stage_events, executor
                )
                for config_index, (config, execution_config) in enumerate(
                    zip(configs, execution_configs)
                )
            )
        )


if __name__ == "__main__":
    args, explicit_args = parse_arguments()

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

    logging.getLogger().setLevel(logging.INFO)
    asyncio.run(main(configs))
