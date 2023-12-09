import os
import re
import logging
import subprocess
from typing import Optional

from .basics import parse_anime_info
from .remove_duplicates import DuplicateRemover


def check_cuda_availability(logger=None):
    if logger is None:
        logger = logging.getLogger()
    try:
        output = subprocess.check_output(
            ["ffmpeg", "-hwaccels"], universal_newlines=True
        )
        return "cuda" in output
    except Exception as e:
        logging.warning(f"Error checking CUDA availability: {e}")
        return False


def get_ffmpeg_command(file, file_pattern, extract_key, logger=None):
    if logger is None:
        logger = logging.getLogger()
    cuda_available = check_cuda_availability(logger)
    command = ["ffmpeg"]

    if cuda_available:
        command.extend(["-hwaccel", "cuda"])
    else:
        logger.warning("CUDA is not available. Proceeding without CUDA.")

    command.extend(["-i", file])

    if extract_key:
        command.extend(["-vf", "select='eq(pict_type,I)'", "-vsync", "vfr"])
    else:
        command.extend(
            [
                "-filter:v",
                "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB",
            ]
        )

    command.extend(["-qscale:v", "1", "-qmin", "1", "-c:a", "copy", file_pattern])

    return command


def extract_and_remove_similar(
    src_dir: str,
    dst_dir: str,
    prefix: Optional[str] = None,
    ep_init: Optional[int] = None,
    extract_key: bool = False,
    duplicate_remover: Optional[DuplicateRemover] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Extracts frames from video files in the specified source directory,
    saves them to the destination directory, and optionally removes similar frames.

    The function supports multiple video file formats such as mp4, mkv, avi, etc.
    It uses FFmpeg to extract frames from videos.
    If a `DuplicateRemover` instance is provided, it removes similar frames within each
    episode's directory and across the entire source directory (for opening and ending).

    Args:
        src_dir (str):
            The directory containing source video files.
        dst_dir (str):
            The directory where extracted frames will be saved.
        prefix (Optional[str]):
            A prefix to add to the names of extracted frames.
            Defaults to None in which case prefix if inferred from file name.
        ep_init (Optional[int]):
            An initial episode number to start from for naming the extracted frames.
            Defaults to None in which case episode number is inferred from file name.
        extract_key (bool):
            Flag indicating whether to extract only key frames.
            Defaults to False.
        duplicate_remover (Optional[DuplicateRemover]):
            An instance of DuplicateRemover to remove duplicate frames.
            Defaults to None in which case no duplicate removal is performed.
        logger (Optional[logging.Logger]):
            A logger for logging messages.
            Defaults to None in which case a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger()
    # Supported video file extensions
    video_extensions = [".mp4", ".mkv", ".avi", ".flv", ".mov", ".wmv"]

    # Recursively find all video files in the specified
    # source directory and its subdirectories
    files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(src_dir)
        for file in files
        if os.path.splitext(file)[1] in video_extensions
    ]

    # Loop through each file
    for i, file in enumerate(sorted(files)):
        # Extract the filename without extension
        filename_without_ext = os.path.splitext(os.path.basename(file))[0]

        # Extract the anime name and episode number
        anime_name, ep_num = parse_anime_info(filename_without_ext)
        anime_name = "_".join(re.split(r"\s+", anime_name))
        prefix_anime = f"{prefix if isinstance(prefix, str) else anime_name}_"
        if isinstance(ep_init, int):
            ep_num = i + ep_init
        elif ep_num is None:
            ep_num = i

        # Create the output directory
        dst_ep_dir = os.path.join(dst_dir, filename_without_ext)
        os.makedirs(dst_ep_dir, exist_ok=True)
        file_pattern = os.path.join(dst_ep_dir, f"{prefix_anime}EP{ep_num}_%d.png")

        # Run ffmpeg on the file, saving the output to the output directory
        ffmpeg_command = get_ffmpeg_command(file, file_pattern, extract_key, logger)
        logger.info(ffmpeg_command)
        subprocess.run(ffmpeg_command, check=True)

        if duplicate_remover is not None:
            duplicate_remover.remove_similar_from_dir(dst_ep_dir)

    # Go through all files again to remove duplicates from op and ed
    if duplicate_remover is not None:
        duplicate_remover.remove_similar_from_dir(dst_dir, portion="first")
        duplicate_remover.remove_similar_from_dir(dst_dir, portion="last")
