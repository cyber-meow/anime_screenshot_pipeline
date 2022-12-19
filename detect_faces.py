import argparse
import cv2
import glob
import os
import json
from tqdm import tqdm

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


def process(args):

    print("loading face detector.")
    detector = create_detector('yolov3')

    print("processing.")
    output_extension = ".png"

    paths = glob.glob(os.path.join(args.src_dir, "*.png"))
    paths = paths + glob.glob(os.path.join(args.src_dir, "*.jpg"))
    paths = paths + glob.glob(os.path.join(args.src_dir, "*.webp"))

    for path in tqdm(paths):
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
                                               crop=(not args.no_cropping),
                                               score_thres=args.score_thres,
                                               ratio_thres=args.ratio_thres,
                                               debug=args.debug)

        idx = 0
        for image, faces_data in zip(images, faces_data_list):
            n_faces = faces_data['n_faces']
            if (args.min_face_number <= n_faces
                    and n_faces <= args.max_face_number):
                _, buf = cv2.imencode(output_extension, image)
                fh_ratio = min(int(faces_data['max_height_ratio'] * 100), 99)
                lb = fh_ratio // args.folder_range * args.folder_range
                ub = lb + args.folder_range
                dst_dir = os.path.join(args.dst_dir, f'{n_faces}_faces')
                dst_dir = os.path.join(dst_dir, f'face_height_ratio_{lb}-{ub}')
                os.makedirs(dst_dir, exist_ok=True)
                with open(
                        os.path.join(dst_dir,
                                     f"{basename}_{idx}{output_extension}"),
                        "wb") as f:
                    buf.tofile(f)
                with open(
                        os.path.join(dst_dir,
                                     f"{basename}_{idx}_facedata.json"),
                        "w") as f:
                    json.dump(faces_data, f)
                idx += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory to load images")
    parser.add_argument("--dst_dir", type=str, help="directory to save images")
    parser.add_argument("--no_cropping",
                        action="store_true",
                        help="do not crop square images around faces")
    parser.add_argument(
        "--min_face_number",
        type=int,
        default=1,
        help="the minimum number of faces an image should contain")
    parser.add_argument(
        "--max_face_number",
        type=int,
        default=10,
        help="the maximum number of faces an image can contain")
    parser.add_argument("--score_thres",
                        type=float,
                        default=0.75,
                        help="score threshold above which is counted as face")
    parser.add_argument("--ratio_thres",
                        type=float,
                        default=2,
                        help="ratioi threshold below which is counted as face")
    parser.add_argument("--folder_range",
                        type=int,
                        default=25,
                        help="the height ratio range of each separate folder")
    parser.add_argument("--debug",
                        action="store_true",
                        help="render rect for face")
    args = parser.parse_args()

    process(args)
