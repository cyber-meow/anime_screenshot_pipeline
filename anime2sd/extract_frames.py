import os
import logging
import subprocess
from typing import Optional

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
    src_dir,
    dst_dir,
    prefix,
    ep_init=1,
    extract_key=False,
    duplicate_remover: Optional[DuplicateRemover] = None,
    logger: Optional[logging.Logger] = None,
):
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

        # Create the output directory
        dst_ep_dir = os.path.join(dst_dir, filename_without_ext)
        os.makedirs(dst_ep_dir, exist_ok=True)
        file_pattern = os.path.join(dst_ep_dir, f"{prefix}EP{i+ep_init}_%d.png")

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
