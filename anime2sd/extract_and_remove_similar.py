import os
import re
import shlex
import logging
from tqdm import tqdm

import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.metrics.pairwise import cosine_similarity

from anime2sd.basics import get_related_paths


def load_image_dataset(dataset_dir):
    """
    Load all images from a directory to create a dataset
    Note that the original dataset does not support .webp
    """
    dataset = fo.Dataset()

    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png',
                        '.bmp', '.gif', '.webp', '.tiff', '.tif']

    # Iterate over all files in the dataset directory
    for root, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            # Check if the file is an image by its extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                filepath = os.path.join(root, filename)

                # Create a sample and add it to the dataset
                sample = fo.Sample(filepath=filepath)
                dataset.add_sample(sample)

    return dataset


def create_dataset_from_subdirs(dataset_dir, portion="first"):
    """
    Create a FiftyOne dataset from a directory of images.

    Args:
        dataset_dir: the directory containing subdirectories of images
        portion:
        either "first" to take the first 1/3 of images from each subdirectory,
        or "last" to take the last 1/3 of images from each subdirectory

    Returns:
        a FiftyOne dataset
    """
    # Create an empty dataset
    dataset = fo.Dataset()

    # Regular expression to extract the number from the filename
    number_extractor = re.compile(r'_(\d+)')

    # Iterate over each subdirectory
    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            # Extract numbers from the filenames and sort them
            image_files = [
                    f for f in os.listdir(subdir_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            image_numbers = [f.split('_')[-1] for f in image_files]
            sorted_files = [x for _, x in sorted(
                zip(image_numbers, image_files))]

            # Determine the threshold based on the largest number
            max_number = max(image_numbers)
            threshold = max_number // 3

            # Depending on the portion, select the files
            if portion == "first":
                selected_files = [
                    f for f in sorted_files if int(number_extractor.search(
                        f).group(1)) <= threshold]
            elif portion == "last":
                selected_files = [
                    f for f in sorted_files if int(number_extractor.search(
                        f).group(1)) > 2 * threshold]
            else:
                raise ValueError("portion must be either 'first' or 'last'")

            # Add these selected images to the dataset
            for image_file in selected_files:
                sample = fo.Sample(
                    filepath=os.path.join(subdir_path, image_file))
                dataset.add_sample(sample)

    return dataset


def mark_duplicate(subdataset, similarity_matrix, thresh=0.985):

    n = len(similarity_matrix)
    similarity_matrix = similarity_matrix - np.identity(n)

    id_map = [s.id for s in subdataset.select_fields(["id"])]
    samples_to_remove = set()
    samples_to_keep = set()

    for idx, sample in enumerate(subdataset):
        max_similarity = similarity_matrix[idx].max()
        sample["max_similarity"] = max_similarity
        sample.save()

    for idx, sample in tqdm(enumerate(subdataset)):
        if sample.id not in samples_to_remove:
            # Keep the first instance of two duplicates
            samples_to_keep.add(sample.id)

            dup_idxs = np.where(similarity_matrix[idx] > thresh)[0]
            for dup in dup_idxs:
                # We kept the first instance so remove all other duplicates
                samples_to_remove.add(id_map[dup])

            if len(dup_idxs) > 0:
                sample.tags.append("has_duplicates")
                sample.save()

        else:
            sample.tags.append("duplicate")
            sample.save()
    return samples_to_remove, samples_to_keep


def remove_similar(dataset, model, thresh=0.985, max_compare_size=10000):
    logging.info('Compute embeddings ...')
    embeddings = dataset.compute_embeddings(model)

    samples_to_remove = set()
    samples_to_keep = set()

    for k in range(0, len(embeddings), max_compare_size):
        end = min(k + max_compare_size, len(embeddings))
        similarity_matrix = cosine_similarity(embeddings[k:end])
        samples_to_remove_sub, samples_to_keep_sub = mark_duplicate(
            dataset[k:end], similarity_matrix, thresh)
        samples_to_remove = samples_to_remove | samples_to_remove_sub
        samples_to_keep = samples_to_keep | samples_to_keep_sub
    logging.info('Removing similar images ...')
    for sample_id in tqdm(samples_to_remove):
        img_path = dataset[sample_id].filepath
        os.remove(img_path)
        related_paths = get_related_paths(img_path)
        for related_path in related_paths:
            if os.path.exists(related_path):
                os.remove(related_path)
    dataset.delete_samples(list(samples_to_remove))


def remove_similar_from_dir(dirpath, model,
                            thresh=0.985, max_cmopare_size=10000):
    dataset = load_image_dataset(dirpath)
    remove_similar(
        dataset, model, thresh=thresh, max_compare_size=max_cmopare_size)


def extract_and_remove_similar(src_dir, dst_dir, prefix,
                               ep_init=1,
                               model_name=None,
                               to_remove_similar=True, thresh=0.985):
    # Supported video file extensions
    video_extensions = ['.mp4', '.mkv', '.avi', '.flv', '.mov', '.wmv']

    # Recursively find all video files in the specified
    # source directory and its subdirectories
    files = [os.path.join(root, file)
             for root, dirs, files in os.walk(src_dir)
             for file in files if os.path.splitext(file)[1]
             in video_extensions]
    if model_name is None:
        model_name = 'mobilenet-v2-imagenet-torch'
    if to_remove_similar:
        model = foz.load_zoo_model(model_name)

    # Loop through each file
    for i, file in enumerate(sorted(files)):
        # Extract the filename without extension
        filename_without_ext = os.path.splitext(os.path.basename(file))[0]

        # Create the output directory
        dst_ep_dir = os.path.join(dst_dir, filename_without_ext)
        os.makedirs(dst_ep_dir, exist_ok=True)
        file_pattern = os.path.join(dst_ep_dir,
                                    f'{prefix}EP{i+ep_init}_%d.png')

        # Run ffmpeg on the file, saving the output to the output directory
        ffmpeg_command = \
            f"ffmpeg -hwaccel cuda -i {shlex.quote(file)} -filter:v "\
            "'mpdecimate=hi=64*200:lo=64*50:"\
            "frac=0.33,setpts=N/FRAME_RATE/TB' "\
            f"-qscale:v 1 -qmin 1 -c:a copy {shlex.quote(file_pattern)}"
        logging.info(ffmpeg_command)
        os.system(ffmpeg_command)

        if to_remove_similar:
            logging.info("Removing duplicates for '{filename_without_ext}':")
            logging.info("Preparing dataset ...")
            dataset = fo.Dataset.from_dir(dst_ep_dir,
                                          dataset_type=fo.types.ImageDirectory)
            remove_similar(dataset, model, thresh=thresh)

    # Go through all files again to remove duplicates from op and ed
    if to_remove_similar:
        logging.info("Removing op duplicates:")
        logging.info("Preparing dataset...")
        dataset = create_dataset_from_subdirs(dst_dir, portion='first')
        remove_similar(dataset, model, thresh=thresh)

        logging.info("Removing ed duplicates:")
        logging.info("Preparing dataset ...")
        dataset = create_dataset_from_subdirs(dst_dir, portion='last')
        remove_similar(dataset, model, thresh=thresh)
