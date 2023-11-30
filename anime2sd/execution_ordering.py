import os
import argparse


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
    ).rstrip(os.path.sep)
    if makedirs:
        os.makedirs(dst_dir, exist_ok=True)
    return dst_dir


def get_src_dir(args, stage):
    """
    Determines the source directory for a given stage of the pipeline.

    Args:
        args (argparse.Namespace):
            The namespace object containing the command-line arguments.
        stage (Union[str, int]): The current stage of the pipeline.

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
    elif stage == "core_tag":
        dst_dir = get_src_dir(args, 5)
        for _ in range(args.compute_core_tag_up_levels):
            dst_dir = os.path.dirname(dst_dir)
        return dst_dir
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


def is_parent_path(path1, path2, both_sides=False):
    parent = os.path.commonpath([path1, path2])
    is_parent = parent == path1
    if both_sides:
        is_parent = is_parent or (parent == path2)
    return is_parent


class ExecutionConfig(object):
    """
    Utilities to order the execution of multiple pipelines when
    multiple configuration files are given.
    """

    def __init__(self, config, configs, execution_configs):
        self.config = config

        self.stage3_dependencies = []
        self.save_core_dependencies = []
        self.stage5_final_dependencies = []
        self.stage6_dependencies = []
        self.stage7_dependencies = []

        self.run_save_core = not self.config.use_existing_core_tag_file
        self.run_stage6 = True
        self.run_stage7 = True

        self.image_types = set([config.image_type])

        for i, config in enumerate(configs):
            if i >= len(execution_configs):
                execution_config_alter = None
            else:
                execution_config_alter = execution_configs[i]
            self.update_dependencies(config, i, execution_config_alter)

        self.image_types = list(self.image_types)

    def update_dependencies(
        self, config_alter, config_alter_index, execution_config_alter
    ):
        # Update stage 3 dependencies to make sure that booru images could be first
        # classified and used to supplement reference images
        if (
            (
                self.config.pipeline_type != "booru"
                or self.config.n_add_to_ref_per_character <= 0
            )
            and config_alter.character_ref_dir == self.config.character_ref_dir
            and config_alter.pipeline_type == "booru"
            and config_alter.n_add_to_ref_per_character > 0
            and config_alter.start_stage <= 3 <= config_alter.end_stage
        ):
            self.stage3_dependencies.append(config_alter_index)

        # Manage logic related to core tag computation and use
        if (
            self.config.start_stage <= 5 <= self.config.end_stage
            and config_alter.start_stage <= 5 <= config_alter.end_stage
        ):
            core_tag_dir = get_src_dir(self.config, "core_tag")
            # To compute core tags, we need to wait for the (first-phase) tagging of all
            # the subdirectories to be completed
            if is_parent_path(
                core_tag_dir, get_src_dir(config_alter, 5), both_sides=True
            ):
                self.save_core_dependencies.append(config_alter_index)
            # If two configs compute core tags from the same directory, we only need to
            # compute core tags once, but we need to wait for this computation to be
            # completed to proceed to final pruning where we read from core tag file
            if core_tag_dir == get_src_dir(config_alter, "core_tag"):
                self.stage5_final_dependencies.append(config_alter_index)
                if self.run_save_core:
                    if execution_config_alter is not None:
                        execution_config_alter.run_save_core = False
                    # Include all related image types for embedding initialization
                    self.image_types.add(config_alter.image_type)

        # Update stage 6 dependencies
        if self.config.start_stage <= 6 <= self.config.end_stage:
            arrange_dir = get_src_dir(self.config, 6)
            # To rearrange images, we need to wait for the tagging of all the
            # subdirectories to be completed
            if (
                config_alter.start_stage <= 5 <= config_alter.end_stage
                and is_parent_path(
                    arrange_dir, get_src_dir(config_alter, 5), both_sides=True
                )
            ):
                self.stage6_dependencies.append(config_alter_index)
            if config_alter.start_stage <= 6 <= config_alter.end_stage:
                arrange_dir_alter = get_src_dir(config_alter, 6)
                # Should only rearrange once if two configs rearrange the same folder
                if arrange_dir_alter == arrange_dir:
                    if execution_config_alter is not None:
                        execution_config_alter.run_stage6 = False
                # Do not rearrange images if this would get rearranged from some
                # parent directory
                elif is_parent_path(arrange_dir_alter, arrange_dir):
                    self.run_stage6 = False

        # Update stage 7 dependencies
        if self.config.start_stage <= 7 <= self.config.end_stage:
            balance_dir = get_src_dir(self.config, 7)
            if config_alter.start_stage <= 6 <= config_alter.end_stage:
                arrange_dir_alter = get_src_dir(config_alter, 6)
                # To compute multiply, we need to wait for the rearrangement of
                # subdiretories to be completed
                if is_parent_path(balance_dir, arrange_dir_alter):
                    self.stage7_dependencies.append(config_alter_index)
                # Do not compute multiply if we get rearranged from some parent folder
                elif is_parent_path(arrange_dir_alter, balance_dir):
                    self.run_stage7 = False
            if config_alter.start_stage <= 7 <= config_alter.end_stage:
                balance_dir_alter = get_src_dir(config_alter, 7)
                # Should only compute multiply once if two configs compute multiplies
                # for the same folder
                if balance_dir_alter == balance_dir:
                    if execution_config_alter is not None:
                        execution_config_alter.run_stage7 = False
                # Do not compute multiply if this would be done from some parent folder
                elif is_parent_path(balance_dir_alter, balance_dir):
                    self.run_stage7 = False


def get_execution_configs(configs):
    execution_configs = []
    for config in configs:
        execution_configs.append(ExecutionConfig(config, configs, execution_configs))
    return execution_configs
