import argparse
import cv2
import os
import json
import shutil

from tqdm import tqdm
from pathlib import Path

import numpy as np
from anime_face_detector import create_detector


def detect_faces(detector,
                 image,
                 crop=True,
                 score_thres=0.75,
                 ratio_thres=2,
                 debug=False):
    preds = detector(image)  # bgr
    h, w = image.shape[:2]
    images = [image]
    faces_data_main = {
        'n_faces': 0,
        'abs_pos': [],
        'rel_pos': [],
        'max_height_ratio': 0,
    }
    faces_data_list = [faces_data_main]
    faces_bbox = []
    faces_bbox_to_crop = []

    for pred in preds:
        bb = pred['bbox']
        score = bb[-1]
        left, top, right, bottom = [int(pos) for pos in bb[:4]]
        fw, fh = right - left, bottom - top
        # ignore the face if too far from square or too low score
        if (fw / fh > ratio_thres or
                fh / fw > ratio_thres or score < score_thres):
            continue
        faces_bbox.append(bb[:4])
        faces_data_main['n_faces'] = faces_data_main['n_faces'] + 1
        left_rel = left / w
        top_rel = top / h
        right_rel = right / w
        bottom_rel = bottom / h
        faces_data_main['abs_pos'].append([left, top, right, bottom])
        faces_data_main['rel_pos'].append(
            [left_rel, top_rel, right_rel, bottom_rel])
        if fh / h > faces_data_main['max_height_ratio']:
            faces_data_main['max_height_ratio'] = fh / h
        if debug:
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255),
                          4)
        # Crop only if the face is not too big
        if max(fw, fh) < min(w, h):
            faces_bbox_to_crop.append(bb[:4])

    # Crop some sqaures in case where the image is not square
    # Potential improvement: we can crop the character with some
    # script that can deteect the character position
    if h != w and crop:
        for face_bbox in faces_bbox_to_crop:
            image_cropped, faces_data = crop_sqaure(image, face_bbox,
                                                    faces_bbox)
            images.append(image_cropped)
            faces_data_list.append(faces_data)

    return images, faces_data_list


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
        'abs_pos': abs_pos,
        'rel_pos': rel_pos,
        'max_height_ratio': max_height_ratio,
    }
    return image, faces_data


def get_npeople_and_characters_from_tags(tags_content):

    number_dictinary = {
        '1girl': 1,
        '1boy': 1,
        '6+girls': 6,
        '6+boys': 6,
    }
    for k in range(2, 6):
        number_dictinary[f'{k}girls'] = k
        number_dictinary[f'{k}boys'] = k

    n_people = 0
    characters = ['unknown']
    for line in tags_content:
        if line.startswith('character:'):
            characters = line.lstrip('character:').split(',')
            characters = [character.strip() for character in characters]
        for key in number_dictinary:
            if key in line:
                n_people += number_dictinary[key]
    return n_people, characters


def update_dst_dir_and_facedata(path, faces_data, dst_dir_base,
                                use_tags, use_character_folder, cropped):
    fh_ratio = min(int(faces_data['max_height_ratio'] * 100), 99)
    lb = fh_ratio // args.folder_range * args.folder_range
    ub = lb + args.folder_range
    face_ratio_folder = f'face_height_ratio_{lb}-{ub}'
    faces_data['characters'] = ['unknown']
    faces_data['cropped'] = cropped
    if use_character_folder:
        parent_folder, character_folder = os.path.split(os.path.dirname(path))
        characters = character_folder.split('+')
        if not cropped:
            # faces_data['n_people'] = len(characters)
            faces_data['characters'] = characters
        # Notice that number of people in dictionary can be further modified by
        # the tag file but as for the folder name we use the number of
        # characters provide in folder names
        n_characters = len(characters)
        dst_dir = os.path.join(args.dst_dir, f'{n_characters}_characters')
        dst_dir = os.path.join(dst_dir, face_ratio_folder)
        dst_dir = os.path.join(dst_dir, character_folder)
    if use_tags and not cropped:
        tags_file = path + '.tags'
        if os.path.exists(tags_file):
            with open(tags_file, 'r') as f:
                lines = f.readlines()
            n_people, characters = get_npeople_and_characters_from_tags(lines)
            if n_people >= 6:
                faces_data['n_people'] = 'many'
            elif n_people > 0:
                faces_data['n_people'] = n_people
            faces_data['characters'] = characters
        else:
            print('Warning: --use_tags specified but tags file '
                  + f'{tags_file} not found')
        if not use_character_folder:
            dst_dir = os.path.join(args.dst_dir, f'{n_people}_people')
            dst_dir = os.path.join(dst_dir, face_ratio_folder)
    elif not use_character_folder:
        n_faces = faces_data['n_faces']
        dst_dir = os.path.join(args.dst_dir, f'{n_faces}_faces')
        dst_dir = os.path.join(dst_dir, face_ratio_folder)
    return dst_dir, faces_data


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def process(args):

    print("loading face detector.")
    detector = create_detector('yolov3')

    print("processing.")
    output_extension = ".png"

    paths = get_files_recursively(args.src_dir)

    for path in tqdm(paths):
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            print(f"image has alpha. ignore: {path}")
            image = image[:, :, :3].copy()

        h, w = image.shape[:2]

        images, faces_data_list = detect_faces(detector,
                                               image,
                                               crop=args.crop,
                                               score_thres=args.score_thres,
                                               ratio_thres=args.ratio_thres,
                                               debug=args.debug)
        tags_file = path + '.tags'

        for idx, (image, facedata) in enumerate(
                zip(images, faces_data_list)):
            dst_dir, facedata = update_dst_dir_and_facedata(
                path, facedata, args.dst_dir,
                args.use_tags, args.use_character_folder, idx != 0)
            n_faces = facedata['n_faces']
            if (not isinstance(n_faces, int)
                    or (args.min_face_number <= n_faces
                        and n_faces <= args.max_face_number)):
                os.makedirs(dst_dir, exist_ok=True)
                new_path_base = os.path.join(dst_dir, basename)
                if idx > 0:
                    new_path_base = f'{new_path_base}_{idx}'
                if idx == 0 and (not args.debug) and args.move_file:
                    ext = os.path.splitext(path)[1]
                    new_path = f'{new_path_base}{ext}'
                    shutil.move(path, new_path)
                else:
                    _, buf = cv2.imencode(output_extension, image)
                    new_path = f'{new_path_base}{output_extension}'
                    with open(new_path, "wb") as f:
                        buf.tofile(f)
                with open(
                        os.path.join(dst_dir,
                                     f"{new_path_base}.facedata.json"),
                        "w") as f:
                    json.dump(facedata, f)
                if idx == 0 and args.use_tags and os.path.exists(tags_file):
                    if args.move_file and (not args.debug):
                        shutil.move(tags_file, new_path + '.tags')
                    else:
                        shutil.copy(tags_file, new_path + '.tags')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str,
                        help="Directory to load images")
    parser.add_argument("--dst_dir", type=str,
                        help="Directory to save images")
    parser.add_argument("--crop",
                        action="store_true",
                        help="Crop square images around faces")
    parser.add_argument(
        "--move_file",
        action="store_true",
        help="Move the orignal image instead of saving a new one")
    parser.add_argument(
        "--use_tags",
        action="store_true",
        help="If provided, write character, number of people " +
        "information using tags and arrange folder per se"
    )
    parser.add_argument(
        "--use_character_folder",
        action="store_true",
        help="If true use character folder to determine number of people " +
        "in the outer folder"
    )
    parser.add_argument(
        "--min_face_number",
        type=int,
        default=1,
        help="The minimum number of faces an image should contain")
    parser.add_argument(
        "--max_face_number",
        type=int,
        default=10,
        help="The maximum number of faces an image can contain")
    parser.add_argument("--score_thres",
                        type=float,
                        default=0.75,
                        help="Score threshold above which is counted as face")
    parser.add_argument("--ratio_thres",
                        type=float,
                        default=2,
                        help="Ratio threshold below which is counted as face")
    parser.add_argument("--folder_range",
                        type=int,
                        default=25,
                        help="The height ratio range of each separate folder")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Render rect for face")
    args = parser.parse_args()

    process(args)
