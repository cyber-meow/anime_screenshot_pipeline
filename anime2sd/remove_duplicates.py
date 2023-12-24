import os
import logging
from tqdm import tqdm
from typing import List, Tuple, Set, Optional
from PIL import Image

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import timm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .basics import get_related_paths, get_images_recursively


class ImageDataset(Dataset):
    """
    A dataset class for loading and transforming images from a directory.
    """

    def __init__(self, image_paths: List[str], transform: callable):
        """
        Initializes the ImageDataset with image paths and a transform.

        Args:
            image_paths (List[str]):
                List of paths to the images.
            transform (callable):
                A function/transform that takes in a PIL image and returns
                a transformed version.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image_path, image

    @classmethod
    def from_directory(cls, dataset_dir: str, transform: callable):
        """
        Creates an ImageDataset from a directory of images.

        Args:
            dataset_dir (str): Path to the directory containing images.
            transform (callable): A function/transform for image processing.

        Returns:
            ImageDataset: The constructed dataset.
        """
        image_paths = get_images_recursively(dataset_dir)
        return cls(image_paths, transform)

    @classmethod
    def from_subdirectories(
        cls, dataset_dir: str, transform: callable, portion: Optional[str] = "first"
    ):
        """
        Creates an ImageDataset from subdirectories of images,
        selecting a specific portion of images.
        This is useful for creating dataset that contain opening and ending of animes.
        This assumes that different episodes are stored in different subfolders,
        and that the extracted frames follow a specific naming convention so
        that their numbers can be extracted.

        Args:
            dataset_dir (str):
                Path to the directory containing subdirectories of images.
            transform (callable):
                A function/transform for image processing.
            portion (Optional[str]):
                Specifies which portion of images to consider ('first' or 'last').

        Returns:
            ImageDataset: The constructed dataset.
        """

        def get_image_number(filename):
            return int(os.path.splitext(filename)[0].split("_")[-1])

        image_paths = []
        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            if os.path.isdir(subdir_path):
                image_files = get_images_recursively(subdir_path)
                image_numbers = [get_image_number(f) for f in image_files]
                sorted_files = [x for _, x in sorted(zip(image_numbers, image_files))]

                max_number = max(image_numbers)
                thresholdold = max_number // 3

                if portion == "first":
                    selected_files = [
                        f for f in sorted_files if get_image_number(f) <= thresholdold
                    ]
                elif portion == "last":
                    selected_files = [
                        f
                        for f in sorted_files
                        if get_image_number(f) > 2 * thresholdold
                    ]
                else:
                    raise ValueError("portion must be either 'first' or 'last'")

                image_paths.extend(selected_files)

        return cls(image_paths, transform)


class DuplicateRemover(object):
    """
    A class to remove duplicate images from a dataset based on image embeddings.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        threshold: float = 0.96,
        max_compare_size: int = 10000,
        dataloader_batch_size: int = 16,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """Initializes the DuplicateRemover object.

        Attributes:
            model_name (str): Name of the model used for generating embeddings.
            device (str): Device to use for computations.
            threshold (float):
                Threshold for cosine similarity to consider images as duplicates.
            max_compare_size (int): Maximum number of images to compare at once.
            dataloader_batch_size (int): Batch size for the DataLoader.
            dataloader_num_workers (int): Number of worker threads for DataLoader.
            pin_memory (bool): Whether to use pinned memory in DataLoader.
            logger (logging.Logger): Logger for logging information.
        """
        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.logger.info(f"Loading {model_name} ...")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.max_compare_size = max_compare_size
        self.dataloader_batch_size = dataloader_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory
        self.data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**self.data_cfg)

    def compute_embeddings(self, dataset: ImageDataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.dataloader_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        """
        Compute embeddings for all images in the dataset.

        Args:
            dataset: Dataset containing images.

        Returns:
            np.ndarray: Array of embeddings of shape n_images x n_featuers.
        """
        embeddings = []

        with torch.no_grad(), torch.autocast(device_type=self.device):
            for _, images in tqdm(dataloader):
                images = images.to(self.device)
                features = self.model(images)
                embeddings.append(features.cpu().float().numpy())

        return np.vstack(embeddings)

    def get_duplicate(
        self, embeddings: np.ndarray, indices: Optional[np.ndarray] = None
    ) -> Tuple[Set[int], Set[int]]:
        """
        Identify duplicate images based on embeddings.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            indices (Optional[np.ndarray]): Indices of embeddings to consider.

        Returns:
            Tuple[Set[int], Set[int]]: Sets of indices to remove and keep.
        """
        if indices is None:
            indices = np.arange(len(embeddings))
        embeddings = embeddings[indices]
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = similarity_matrix - np.identity(len(similarity_matrix))

        samples_to_remove = set()
        samples_to_keep = set()

        for idx in tqdm(range(len(embeddings))):
            sample_id = indices[idx]
            if sample_id not in samples_to_remove:
                # Keep the first instance of two duplicates
                samples_to_keep.add(sample_id)

                dup_idxs = np.where(similarity_matrix[idx] > self.threshold)[0]
                for dup in dup_idxs:
                    # We kept the first instance so remove all other duplicates
                    samples_to_remove.add(indices[dup])
        return samples_to_remove, samples_to_keep

    def remove_similar(self, dataset: ImageDataset):
        """
        Remove similar images from the dataset.

        Args:
            dataset: Dataset containing images.
        """
        self.logger.info(f"Compute embeddings for {len(dataset)} images ...")
        embeddings = self.compute_embeddings(dataset)

        samples_to_remove = set()

        for k in range(0, len(embeddings), self.max_compare_size):
            end = min(k + self.max_compare_size, len(embeddings))
            samples_to_remove_sub, _ = self.get_duplicate(
                embeddings, indices=np.arange(k, end)
            )
            samples_to_remove = samples_to_remove | samples_to_remove_sub

        self.logger.info("Removing similar images ...")
        for sample_id in tqdm(samples_to_remove):
            img_path, _ = dataset[sample_id]
            os.remove(img_path)
            related_paths = get_related_paths(img_path)
            for related_path in related_paths:
                if os.path.exists(related_path):
                    os.remove(related_path)

    def remove_similar_from_dir(self, dirpath: str, portion: Optional[str] = None):
        """
        Remove similar images from a directory.

        Args:
            dirpath (str):
                Path to the directory containing images.
            portion (Optional[str]):
                Specifies which portion of images to consider ('first' or 'last').
                Useful for removing duplicates from opening and ending of animes.
                This assumes that different episodes are stored in different subfolders,
                and that the extracted frames follow a specific naming convention so
                that their numbers can be extracted.
        """
        if portion:
            rtype = "op" if portion == "first" else "ed"
            self.logger.info(f"Removing {rtype} duplicates for '{dirpath}' ...")
            dataset = ImageDataset.from_subdirectories(
                dirpath, self.transform, portion=portion
            )
        else:
            self.logger.info(f"Removing duplicates for '{dirpath}' ...")
            dataset = ImageDataset.from_directory(dirpath, self.transform)
        if len(dataset) == 0:
            return
        self.remove_similar(dataset)
