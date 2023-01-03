import argparse
import cv2
import os
import json

from tqdm import tqdm
from pathlib import Path

import numpy as np


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


# Crop images to contain a certain face
def crop_sqaure(image, face_bbox, faces_bbox, debug=False):
    h, w = image.shape[:2]
    left, top, right, bottom = [int(pos) for pos in face_bbox]
    # crop to the largest sqaure
    crop_size = min(h, w)
    n_faces = 0
    abs_pos = []
    rel_pos = []
    max_height_ratio = 0
    # paysage
    if h < w:
        # Put face in the middle, horizontally
        cx = int((left + right) / 2)
        left_crop = max(cx - crop_size // 2, 0)
        right_crop = left_crop + crop_size
        if right_crop > w:
            right_crop = w
            left_crop = right_crop - crop_size
        image = image[:, left_crop:right_crop]
        # Find faces mostly (more than 60%) contained in the cropped image
        for bb in faces_bbox:
            left, top, right, bottom = [int(pos) for pos in bb[:4]]
            cx = (left + right) / 2
            fw = right - left
            left_tight = cx - fw * 0.1
            right_tight = cx + fw * 0.1
            if left_tight >= left_crop and right_tight <= right_crop:
                n_faces += 1
                left = left - left_crop
                right = right - left_crop
                left_rel = left / crop_size
                top_rel = top / crop_size
                right_rel = right / crop_size
                bottom_rel = bottom / crop_size
                abs_pos.append([left, top, right, bottom])
                rel_pos.append([left_rel, top_rel, right_rel, bottom_rel])
                fh = bottom - top
                if fh / crop_size > max_height_ratio:
                    max_height_ratio = fh / h
                if debug:
                    cv2.rectangle(image, (left, top), (right, bottom),
                                  (255, 0, 255), 4)
    # portrait
    if h > w:
        # Try to put the head including hair at the top
        fh = bottom - top
        top_crop = max(top - int(fh // 2), 0)
        bottom_crop = top_crop + crop_size
        if bottom_crop > h:
            bottom_crop = h
            top_crop = bottom_crop - crop_size
        image = image[top_crop:bottom_crop]
        # Find faces mostly (more than 60%) contained in the cropped image
        for bb in faces_bbox:
            left, top, right, bottom = [int(pos) for pos in bb[:4]]
            cy = (top + bottom) / 2
            fh = bottom - top
            top_tight = cy - fh * 0.1
            bottom_tight = cy + fh * 0.1
            if top_tight >= top_crop and bottom_tight <= bottom_crop:
                n_faces += 1
                top = top - top_crop
                bottom = bottom - top_crop
                left_rel = left / crop_size
                top_rel = top / crop_size
                right_rel = right / crop_size
                bottom_rel = bottom / crop_size
                abs_pos.append([left, top, right, bottom])
                rel_pos.append([left_rel, top_rel, right_rel, bottom_rel])
                fh = bottom - top
                if fh / crop_size > max_height_ratio:
                    max_height_ratio = fh / h
                if debug:
                    cv2.rectangle(image, (left, top), (right, bottom),
                                  (255, 0, 255), 4)
    if h == w:
        raise Exception(
            'This function should only be called for non-square images')
    faces_data = {
        'n_faces': n_faces,
        'facepos': rel_pos,
        'fh_ratio': max_height_ratio,
        'cropped': True,
    }
    return image, faces_data


def main(args):

    print('processing.')
    output_extension = '.png'

    paths = get_files_recursively(args.src_dir)

    for path in tqdm(paths):
        # print(path)
        path_noext = os.path.splitext(path)[0]

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
            # print(f'image has alpha. ignore: {path}')
            image = image[:, :, :3].copy()

        h, w = image.shape[:2]
        # No augmentation if the image is square
        if h == w:
            continue
        try:
            with open(f'{path_noext}.json', 'r') as f:
                facedata = json.load(f)
        except FileNotFoundError:
            print(f'facedata of {path} not found, skip')
            continue
        if facedata['n_faces'] < args.min_face_number:
            continue

        faces_bbox = []
        for rel_pos in facedata['facepos']:
            left, top, right, bottom = rel_pos
            faces_bbox.append(
                [left*w, top*h, right*w, bottom*h])
        for idx, bbox in enumerate(faces_bbox):
            left, top, right, bottom = bbox
            fw = right - left
            fh = bottom - top
            # No crop if the face is too large
            if max(fw, fh) >= min(w, h):
                continue
            image_cropped, facedata_cropped = crop_sqaure(
                image, bbox, faces_bbox)
            new_path_noext = f'{os.path.splitext(path)[0]}_{idx+1}'
            _, buf = cv2.imencode(output_extension, image_cropped)
            new_path = f'{new_path_noext}{output_extension}'
            with open(new_path, 'wb') as f:
                buf.tofile(f)
            with open(f'{new_path_noext}.json', 'w') as f:
                json.dump(facedata_cropped, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir', type=str,
        help='Directory to load images')
    parser.add_argument(
        '--min_face_number',
        type=int,
        default=2,
        help='Crop if The number of faces in image larger than this number')
    args = parser.parse_args()

    main(args)
