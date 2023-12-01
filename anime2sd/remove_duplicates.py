import os
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import timm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .basics import get_related_paths, get_images_recursively


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image_path, image

    @classmethod
    def from_directory(cls, dataset_dir, transform):
        image_paths = get_images_recursively(dataset_dir)
        return cls(image_paths, transform)

    @classmethod
    def from_subdirectories(cls, dataset_dir, transform, portion="first"):
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


def remove_similar_from_dir(
    dirpath,
    model_name,
    threshold=0.985,
    max_cmopare_size=10000,
    device="cuda",
    logger=None,
):
    if logger is None:
        logger = logging.getLogger()
    model = timm.create_model(model_name)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    dataset = ImageDataset.from_directory(dirpath, transform=transform)
    remove_similar(
        dataset,
        model,
        threshold=threshold,
        max_compare_size=max_cmopare_size,
        device=device,
        logger=logger,
    )


class DuplicateRemover(object):
    def __init__(
        self,
        model_name,
        device=None,
        threshold=0.96,
        max_compare_size=10000,
        dataloader_batch_size=16,
        dataloader_num_workers=4,
        pin_memory=True,
        logger=None,
    ):
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

    def compute_embeddings(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.dataloader_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        embeddings = []

        with torch.no_grad(), torch.autocast(device_type=self.device):
            for _, images in tqdm(dataloader):
                images = images.to(self.device)
                features = self.model(images)
                embeddings.append(features.cpu().numpy())

        print(np.vstack(embeddings).shape)
        # n_images x n_featuers
        return np.vstack(embeddings)

    def get_duplicate(self, embeddings, indices=None):
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

    def remove_similar(self, dataset):
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

    def remove_similar_from_dir(self, dirpath, portion=None):
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
