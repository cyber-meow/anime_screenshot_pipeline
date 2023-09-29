import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
from anime_face_detector import create_detector


# Each image should contain only one person
def detect_face_and_crop(detector,
                         image,
                         face_crop_aug=1.5,
                         score_thres=0.7,
                         ratio_thres=1.5):
    preds = detector(image)  # bgr
    h, w = image.shape[:2]

    for pred in preds:
        bb = pred['bbox']
        score = bb[-1]
        left, top, right, bottom = bb[:4]
        fw, fh = right - left, bottom - top
        if score > score_thres and max(fw / fh, fh / fw) <= ratio_thres:
            image = crop_face(image, bb[:4], face_crop_aug)
            return image
    return None


# Written by chatgpt
def pad_image_to_square(image):
    height, width = image.shape[:2]

    # Calculate the padding values
    top, bottom = 0, 0
    left, right = 0, 0
    if height > width:
        diff = height - width
        left = diff // 2
        right = diff - left
    elif width > height:
        diff = width - height
        top = diff // 2
        bottom = diff - top
    else:
        # Image is already square, so no padding is needed
        return image

    # Create a black image with the same type as the input image
    # with the calculated padding
    padded_image = cv2.copyMakeBorder(image,
                                      top,
                                      bottom,
                                      left,
                                      right,
                                      cv2.BORDER_CONSTANT,
                                      value=0)

    return padded_image


def crop_face(image, face_bbox, face_crop_aug):
    h, w = image.shape[:2]
    left, top, right, bottom = face_bbox
    fw, fh = right - left, bottom - top
    if max(fw, fh) > min(w, h):
        return pad_image_to_square(image)
    crop_size = min(h, w, int(max(fh, fw) * face_crop_aug))
    # crop_size = min(h, w, max(int(fw * 1.4), int(fh * 1.6)))
    # Put face in the middle, horizontally
    cx = int((left + right) / 2)
    left_crop = max(cx - crop_size // 2, 0)
    right_crop = left_crop + crop_size
    if right_crop > w:
        right_crop = w
        left_crop = right_crop - crop_size
    image = image[:, left_crop:right_crop]
    top_crop = max(int(top) - int(fh // 2), 0)
    bottom_crop = top_crop + crop_size
    if bottom_crop > h:
        bottom_crop = h
        top_crop = bottom_crop - crop_size
    image = image[top_crop:bottom_crop]
    return image


def get_image_files(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
        '*.[Gg][Ii][Ff]', '*.[Ww][Ee][Bb][Pp]'
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).glob(pattern)
    ]

    return image_path_list


def main(args):

    print("loading face detector...")
    detector = create_detector('yolov3')

    output_extension = ".png"

    for dirname in os.listdir(args.src_dir):
        src_dir_path = os.path.join(args.src_dir, dirname)
        if not os.path.isdir(src_dir_path):
            continue
        print(f"processing class {dirname}...")
        image_paths = get_image_files(src_dir_path)
        if args.remove_dir_number:
            dirname = "_".join(dirname.split("_")[1:])
        dst_dir_path = os.path.join(args.dst_dir, dirname)
        os.makedirs(dst_dir_path, exist_ok=True)

        for path in tqdm(image_paths):
            basename = os.path.splitext(os.path.basename(path))[0]
            image = cv2.imdecode(np.fromfile(path, np.uint8),
                                 cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                # print(f"image has alpha. ignore: {path}")
                image = image[:, :, :3].copy()

            image_cropped = detect_face_and_crop(
                detector,
                image,
                face_crop_aug=args.face_crop_aug,
                score_thres=args.score_thres,
                ratio_thres=args.ratio_thres)
            # No face detected
            if image_cropped is None:
                print(f"No face is detected in {path}")
                if not args.keep_no_face_image:
                    continue
                image_cropped = image
            _, buf = cv2.imencode(output_extension, image_cropped)
            with open(
                    os.path.join(dst_dir_path,
                                 f"{basename}{output_extension}"), "wb") as f:
                buf.tofile(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory to load images")
    parser.add_argument("--dst_dir",
                        type=str,
                        help="directory to save images",
                        default="faces/data")
    parser.add_argument("--score_thres",
                        type=float,
                        default=0.5,
                        help="score threshold above which is counted as face")
    parser.add_argument("--ratio_thres",
                        type=float,
                        default=2,
                        help="ratioi threshold below which is counted as face")
    parser.add_argument(
        "--remove_dir_number",
        action="store_true",
        help="remove number from the directory with name 'XX_XXX'")
    parser.add_argument(
        "--keep_no_face_image",
        action="store_true",
        help="save directly the images in which face is not detected")
    parser.add_argument(
        '--face_crop_aug',
        type=float,
        default=1.5,
        help='Ratio between size of the cropped image and that of the face')
    args = parser.parse_args()

    main(args)
