# tests/classif/test_classify_characters.py
import pytest
import os
import shutil
from anime2sd.classif.classify_characters import classify_from_directory


@pytest.fixture(scope="module")
def test_data():
    # Define your test directories
    root_dir = "data"
    src_dir = os.path.join(root_dir, "intermediate", "screenshots", "cropped_mini")
    dst_dir = os.path.join(root_dir, "intermediate", "screenshots", "classified_mini")
    os.makedirs(dst_dir, exist_ok=True)
    character_ref_dir = os.path.join(root_dir, "ref_images")

    # (Optional) Setup the test data before running the tests
    # shutil.copytree('path_to_sample_data', src_dir)
    # shutil.copytree('path_to_sample_ref_images', character_ref_dir)

    return src_dir, dst_dir, character_ref_dir


def test_clustering(test_data):
    src_dir, dst_dir, _ = test_data
    # Call the function with the test arguments
    classify_from_directory(
        src_dir,
        dst_dir,
        None,
        to_extract_from_noise=True,
        keep_unnamed=True,
        clu_min_samples=5,
        merge_threshold=0.85,
        move=False,
    )


def test_classify_ref(test_data):
    src_dir, dst_dir, character_ref_dir = test_data
    # Call the function with the test arguments
    classify_from_directory(
        src_dir,
        dst_dir,
        character_ref_dir,
        to_extract_from_noise=True,
        keep_unnamed=True,
        clu_min_samples=5,
        merge_threshold=0.85,
        move=False,
    )
