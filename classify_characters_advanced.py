import os
import argparse
import json

import cv2
import numpy as np
from PIL import Image

from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.models import load_model


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


def get_head_images(image, facedata, face_crop_aug):
    h, w = image.shape[:2]
    faces_bbox = []
    for rel_pos in facedata['facepos']:
        left, top, right, bottom = rel_pos
        faces_bbox.append(
            [left*w, top*h, right*w, bottom*h])
    head_images = []
    if len(faces_bbox) > 1:
        face_crop_aug = 1.5
    for bbox in faces_bbox:
        head_images.append(get_head_image(image, bbox, face_crop_aug))
    return head_images


def get_characters(head_images, model_cls, class_names, cls_thresh):

    characters = []
    images_new = []

    for img in head_images:
        img = img[:, :, ::-1]       # RGB -> BGR
        image_size = 448
        size = max(img.shape[0:2])
        interp = cv2.INTER_AREA if size > image_size else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (image_size, image_size), interpolation=interp)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        img = img.astype(np.float32)
        images_new.append(img)
    probs = (model_cls(np.array(images_new), training=False
                       ).numpy().astype(float))
    idxs = np.argmax(probs, axis=1)
    for idx, prob in zip(idxs, probs):
        prob = prob[idx]
        if prob > cls_thresh:
            class_name = class_names[idx]
            # if class_name == 'ood':
            #     class_name = 'unknown'
        else:
            class_name = 'unknown'
        characters.append(class_name)
    # idxs = np.argsort(probs, axis=-1)
    # save_num = 8
    # for idx, prob in zip(idxs, probs):
    #     characters.append(
    #         [(class_names[idx[-k]], prob[idx[-k]])
    #          for k in range(1, save_num+1)])
    return characters


def get_tags(imgs, model_tag, tags_all, thresh):
    imgs_new = []
    # for wd1.4 tagger
    for img in imgs:
        img = img[:, :, ::-1]       # RGB -> BGR
        image_size = 448
        size = max(img.shape[0:2])
        interp = cv2.INTER_AREA if size > image_size else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (image_size, image_size), interpolation=interp)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        img = img.astype(np.float32)
        imgs_new.append(img)
    probs = model_tag(np.array(imgs_new), training=False)
    tags = []

    for prob in probs:
        tags_current = []
        effective = np.nonzero(prob[4:] >= thresh)[0]
        for i in effective:
            tags_current.append(tags_all[i])
        # The following for is extremely slow
        # for i, p in enumerate(prob[4:]):
        #     if p >= thresh:
        #         tags_current.append(tags_all[i])
        tags.append(tags_current)
    return tags


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


def get_head_image(image, face_bbox, face_crop_aug=1.5):
    h, w = image.shape[:2]
    left, top, right, bottom = face_bbox
    fw, fh = right - left, bottom - top
    if max(fw, fh) > min(w, h):
        return pad_image_to_square(image)
    # crop_size = min(h, w, max(int(fw * 1.4), int(fh * 1.6)))
    crop_size = min(h, w, int(max(fh, fw) * face_crop_aug))
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


def main(args):

    classname_file = os.path.join(args.checkpoint_dir, 'classnames.txt')

    with open(classname_file, 'r') as f:
        classnames = [line.strip() for line in f.readlines()]

    print('Loading classifier...')

    model = load_model(args.checkpoint_dir)
    # if args.checkpoint_aux_dir is not None:
    #     model_aux = load_model(args.checkpoint_aux_dir)

    file_list = get_files_recursively(args.src_dir)

    file_path_batch = []
    head_image_batch = []
    file_character_dict = dict()

    print('Processing...')
    for idx, file_path in enumerate(tqdm(file_list)):

        file_character_dict[file_path] = []

        # image = cv2.imdecode(np.fromfile(file_path, np.uint8),
        #                      cv2.IMREAD_UNCHANGED)
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        filename_noext = os.path.splitext(file_path)[0]

        json_file = filename_noext + '.json'
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f'Error: {json_file} not found')
            exit(1)

        if 'character' in metadata and not args.overwrite:
            print(f'Warning: attribute `character` found in {json_file}, ' +
                  'skip')
            continue

        head_images = get_head_images(image, metadata, args.face_crop_aug)

        while len(head_images) > 0:
            file_path_batch.append(file_path)
            head_image_batch.append(head_images.pop(0))
            if len(head_image_batch) == args.batch_size:
                characters = get_characters(
                    head_image_batch, model, classnames, args.cls_thresh)
                for file_path, character in zip(file_path_batch, characters):
                    file_character_dict[file_path].append(character)
                file_path_batch = []
                head_image_batch = []

        if (idx + 1) % args.save_frequency == 0 or idx == len(file_list)-1:
            if len(head_image_batch) > 0:
                characters = get_characters(
                    head_image_batch, model, classnames, args.cls_thresh)
                for file_path, character in zip(file_path_batch, characters):
                    file_character_dict[file_path].append(character)
            for file_path in file_character_dict:
                filename_noext = os.path.splitext(file_path)[0]
                json_file = filename_noext + '.json'
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                characters = file_character_dict[file_path]
                metadata['character'] = characters
                with open(json_file, "w") as f:
                    json.dump(metadata, f)
            file_path_batch = []
            head_image_batch = []
            file_character_dict = dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True,
                        help='Source directory of images')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size for inference')
    parser.add_argument(
        '--save_frequency',
        type=int,
        default=100,
        help='set to n so that metadata are saved every n images')
    parser.add_argument(
        '--face_crop_aug',
        type=float,
        default=3.2,
        help='Ratio between size of the cropped image and that of the face')
    parser.add_argument(
        '--image_size',
        default=448,
        type=int,
        help='Image (square) resolution size')
    parser.add_argument(
        '--cls_thresh',
        type=float,
        default=0.5,
        help="threshold of confidence to classify as character")
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing character metadata')
    args = parser.parse_args()

    main(args)
