import os
import csv
import time
import logging
import fnmatch
from tqdm import tqdm

import numpy as np

from anime2sd.basics import get_images_recursively


class WeightTree(object):
    def __init__(self, dirname, weight_mapping=None, progress_bar=None):
        self.dirname = dirname
        self.n_images = 0
        self.contain_images = False
        self.children = []

        for path in os.listdir(dirname):
            path = os.path.join(self.dirname, path)
            if os.path.isfile(path):
                extension = os.path.splitext(path)[1]
                if extension.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    if progress_bar is not None:
                        progress_bar.update(1)
                    self.n_images += 1
                    self.contain_images = True
            elif os.path.isdir(path):
                sub_weight_tree = WeightTree(path, weight_mapping, progress_bar)
                if sub_weight_tree.contain_images or len(sub_weight_tree.children) > 0:
                    self.children.append(sub_weight_tree)
        self.weight = self.modify_weight(weight_mapping)

    def modify_weight(self, training_weights):
        if training_weights is None:
            return 1
        basename = os.path.basename(self.dirname)
        if basename in training_weights:
            # print(self.dirname)
            # print(training_weights[basename])
            return float(training_weights[basename])
        for pattern in training_weights:
            if fnmatch.fnmatch(self.dirname, pattern):
                # print(self.dirname)
                # print(training_weights[pattern])
                return float(training_weights[pattern])
        return 1

    def compute_sampling_prob(self, baseprob, dir_list, prob_list, n_images_list):
        weights_list = []
        for weight_tree in self.children:
            weights_list.append(weight_tree.weight)
        if self.contain_images:
            weights_list.append(self.weight)
        probs = np.array(weights_list) / np.sum(weights_list)
        # Modify dir_list and prob_list in place
        if self.contain_images:
            dir_list.append(self.dirname)
            prob_list.append(baseprob * probs[-1])
            n_images_list.append(self.n_images)
        for i, weight_tree in enumerate(self.children):
            weight_tree.compute_sampling_prob(
                baseprob * probs[i], dir_list, prob_list, n_images_list
            )


def read_weight_mapping(weight_mapping_csv):
    weight_mapping = {}
    with open(weight_mapping_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            pattern, weight = row
            weight_mapping[pattern] = weight
    return weight_mapping


def get_repeat(
    src_dir, weight_mapping, min_multiply=1, max_multiply=100, log_file=None
):
    n_images_totol = len(get_images_recursively(src_dir))
    bar = tqdm(total=n_images_totol)

    weight_tree = WeightTree(src_dir, weight_mapping, bar)

    dir_list = []
    prob_list = []
    n_images_list = []

    weight_tree.compute_sampling_prob(1, dir_list, prob_list, n_images_list)

    probs = np.array(prob_list)
    n_images_array = np.array(n_images_list)
    per_image_weights = probs / n_images_array

    # This makes the weights larger than 1
    per_image_multiply = per_image_weights / np.min(per_image_weights)
    per_image_multiply = per_image_multiply * min_multiply
    per_image_multiply_final = np.minimum(
        np.around(per_image_multiply, 2), max_multiply
    )

    if log_file is not None:
        logger = logging.getLogger()
        original_handlers = logger.handlers[:]
        for handler in original_handlers:
            logger.removeHandler(handler)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w")

    n_images_total = 0
    n_images_virtual_total = 0

    for k in np.argsort(per_image_multiply):
        dirname = dir_list[k]
        n_images = n_images_list[k]
        multiply = per_image_multiply_final[k]
        n_images_total += n_images
        n_images_virtual_total += n_images * multiply
        with open(os.path.join(dirname, "multiply.txt"), "w") as f:
            f.write(str(multiply))
        if log_file is not None:
            logging.info(dirname)
            logging.info(f"sampling probability: {prob_list[k]}")
            logging.info(f"number of images: {n_images}")
            logging.info(f"original multipy: {per_image_multiply[k]}")
            logging.info(f"final multipy: {multiply}\n")

    logging.info(f"Number of images: {n_images_totol}")
    logging.info(f"Virtual dataset size: {n_images_virtual_total}")
    time.sleep(1)

    if log_file is not None:
        logger.handlers = []  # Removing existing handlers
        for handler in original_handlers:
            logger.addHandler(handler)
        logging.info(f"Number of images: {n_images_totol}")
        logging.info(f"Virtual dataset size: {n_images_virtual_total}")
        time.sleep(1)
