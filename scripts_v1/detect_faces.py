import argparse
import cv2
import os
import json

from tqdm import tqdm
from pathlib import Path

import numpy as np
from anime_face_detector import create_detector


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
        '*.[Gg][Ii][Ff]', '*.[Ww][Ee][Bb][Pp]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def detect_faces(detector,
                 image,
                 score_thres=0.75,
                 ratio_thres=2,
                 debug=False):
    preds = detector(image)  # bgr
    h, w = image.shape[:2]
    facedata = {
        'n_faces': 0,
        'facepos': [],
        'fh_ratio': 0,
        'cropped': False,
    }

    for pred in preds:
        bb = pred['bbox']
        score = bb[-1]
        left, top, right, bottom = [int(pos) for pos in bb[:4]]
        fw, fh = right - left, bottom - top
        # ignore the face if too far from square or too low score
        if (fw / fh > ratio_thres or
                fh / fw > ratio_thres or score < score_thres):
            continue
        facedata['n_faces'] = facedata['n_faces'] + 1
        left_rel = left / w
        top_rel = top / h
        right_rel = right / w
        bottom_rel = bottom / h
        facedata['facepos'].append(
            [left_rel, top_rel, right_rel, bottom_rel])
        if fh / h > facedata['fh_ratio']:
            facedata['fh_ratio'] = fh / h
        if debug:
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255),
                          4)

    return facedata


def main(args):

    print("loading face detector.")
    detector = create_detector('yolov3')

    print("processing.")

    paths = get_files_recursively(args.src_dir)

    for path in tqdm(paths):
        # print(path)
        filename_noext = os.path.splitext(path)[0]

        try:
            image = cv2.imdecode(
                np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        except cv2.error as e:
            print(f'Error reading the image {path}: {e}')
            continue
        if image is None:
            print(f'Error reading the image {path}: get None')
            continue
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            # print(f"image has alpha. ignore: {path}")
            image = image[:, :, :3].copy()

        h, w = image.shape[:2]

        facedata = detect_faces(detector,
                                image,
                                score_thres=args.score_thres,
                                ratio_thres=args.ratio_thres,
                                debug=args.debug)

        json_file = f"{filename_noext}.json"
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                metadata = json.load(f) | facedata
        else:
            metadata = facedata

        with open(json_file, "w") as f:
            json.dump(metadata, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str,
        help="Directory to load images")
    parser.add_argument(
        "--score_thres",
        type=float,
        default=0.75,
        help="Score threshold above which is counted as face")
    parser.add_argument(
        "--ratio_thres",
        type=float,
        default=2,
        help="Ratio threshold below which is counted as face")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Render rect for face")
    args = parser.parse_args()

    main(args)
