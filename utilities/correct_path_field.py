import argparse
import os
import json
from tqdm import tqdm


from anime2sd.basics import get_corr_meta_names, get_images_recursively


def find_corresponding_image(file_path, ref_images):
    file_path, file_ext = os.path.splitext(file_path)
    new_file_path = "_".join(file_path.split("_")[:-1]) + file_ext
    for img_path in ref_images:
        if os.path.basename(img_path) == os.path.basename(new_file_path):
            return img_path
    return None


def update_metadata(src_dir, ref_dir):
    src_images = get_images_recursively(src_dir)
    ref_images = get_images_recursively(ref_dir)

    for img_path in tqdm(src_images):
        meta_path, _ = get_corr_meta_names(img_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as file:
                metadata = json.load(file)

            corresponding_img_path = find_corresponding_image(img_path, ref_images)
            if corresponding_img_path:
                metadata["path"] = corresponding_img_path
                with open(meta_path, "w") as file:
                    json.dump(metadata, file, indent=4)
            else:
                print(f"Warning: No corresponding image found for {img_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Update image metadata with reference directory paths."
    )
    parser.add_argument(
        "--src_dir", required=True, help="Source directory containing image files."
    )
    parser.add_argument(
        "--ref_dir", required=True, help="Reference directory to match image files."
    )

    args = parser.parse_args()

    update_metadata(args.src_dir, args.ref_dir)


if __name__ == "__main__":
    main()
