import os
import shutil
import argparse
import json

import cv2
from PIL import Image
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange

from vit_animesion import ViT, ViTConfigExtended, PRETRAINED_CONFIGS


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
        '*.[Gg][Ii][Ff]'
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def get_characters(
        image,
        facedata,
        model,
        classid_classname_dic,
        image_size,
        device,
        score_thres=0.6):

    faces_bbox = facedata['abs_pos']
    characters = []

    with torch.no_grad():
        for bbox in faces_bbox:
            head_image = get_head_image(image, bbox)
            head_image = prepare_image(head_image, image_size, device)
            out_cls = model(head_image).squeeze(0)
            idx = torch.argmax(out_cls).cpu().item()
            prob = torch.softmax(out_cls, -1)[idx].item()
            if prob > score_thres:
                class_name = classid_classname_dic.loc[
                    classid_classname_dic['class_id'] == idx,
                    'class_name'].item()
            else:
                class_name = 'unknown'
            characters.append(class_name)
    return characters


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


def get_head_image(image, face_bbox):
    h, w = image.shape[:2]
    left, top, right, bottom = face_bbox
    fw, fh = right - left, bottom - top
    if max(fw, fh) > min(w, h):
        return pad_image_to_square(image)
    crop_size = min(h, w, max(int(fw * 1.4), int(fh * 1.6)))
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


def prepare_image(image, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).to(device).unsqueeze(0)

    return image


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()

        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.num_classes = args.num_classes
        self.configuration.image_size = args.image_size
        self.configuration.max_text_seq_len = 0

        base_model = ViT(self.configuration,
                         name=args.model_name,
                         pretrained=args.pretrained,
                         load_fc_layer=False,
                         ret_interm_repr=args.interm_features_fc,
                         multimodal=args.multimodal,
                         ret_attn_scores=args.ret_attn_scores)
        self.model = base_model

        if args.interm_features_fc:
            self.inter_class_head = nn.Sequential(
                nn.Linear(self.configuration.num_hidden_layers, 1),
                Rearrange(' b d 1 -> b d'), nn.ReLU(),
                nn.LayerNorm(self.configuration.hidden_size,
                             eps=self.configuration.layer_norm_eps),
                nn.Linear(self.configuration.hidden_size,
                          self.configuration.num_classes))
        else:  # original cls head but also doing mlm
            self.class_head = nn.Sequential(
                nn.LayerNorm(self.configuration.hidden_size,
                             eps=self.configuration.layer_norm_eps),
                nn.Linear(self.configuration.hidden_size,
                          self.configuration.num_classes))

    def forward(self, images, text=None, mask=None):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            images (tensor): `b,c,fh,fw`
            text (tensor): b, max_text_seq_len
            mask (bool tensor): (B(batch_size) x S(seq_len))
        """

        exclusion_loss = 0

        if hasattr(self, 'inter_class_head'):
            features, interm_features = self.model(images, text, mask)
        elif hasattr(self, 'class_head'):
            features = self.model(images, text, mask)
        else:
            logits = self.model(images, text)

        if hasattr(self, 'inter_class_head'):
            if hasattr(self, 'exclusion_loss'):
                for i in range(len(interm_features) - self.exc_layers_dist):
                    exclusion_loss += self.exclusion_loss(
                        F.log_softmax(interm_features[i][:, 0, :] /
                                      self.temperature,
                                      dim=1),
                        F.softmax(
                            interm_features[i +
                                            self.exc_layers_dist][:, 0, :]
                            / self.temperature,
                            dim=1))
            interm_features = torch.stack(interm_features, dim=-1)
            logits = self.inter_class_head(interm_features[:, 0])
        elif hasattr(self, 'class_head'):
            logits = self.class_head(features[:, 0])

        return logits


def main(args):

    print('Loading model...')

    model = VisionTransformer(args)
    state_dict = torch.load(args.checkpoint_path,
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    file_list = get_files_recursively(args.src_dir)

    model.eval()

    print('Processing...')
    for file_path in tqdm(file_list):

        image = cv2.imdecode(np.fromfile(file_path, np.uint8),
                             cv2.IMREAD_UNCHANGED)
        basename = os.path.splitext(file_path)[0]

        json_file = basename + '_facedata.json'
        with open(json_file, 'r') as f:
            facedata = json.load(f)
        os.remove(json_file)

        characters = get_characters(
            image, facedata, model,
            classid_classname_dic, args.image_size, device)
        dirname, filename = os.path.split(file_path)
        character_folder = '+'.join(sorted(characters))
        dst_dir = os.path.join(dirname, character_folder)
        os.makedirs(dst_dir, exist_ok=True)
        new_file_path = os.path.join(dst_dir, filename)
        shutil.move(file_path, new_file_path)

        facedata['characters'] = characters
        basename = os.path.splitext(filename)[0]
        json_file = os.path.join(dst_dir, f"{basename}_facedata.json")
        with open(json_file, "w") as f:
            json.dump(facedata, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        help='Source directory of images')
    parser.add_argument(
        '--dataset_path',
        help='Path for the dataset. For classifier id correspondance.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--image_size',
                        default=128,
                        type=int,
                        help='Image (square) resolution size')
    args = parser.parse_args()

    classid_classname_dic = pd.read_csv(os.path.join(args.dataset_path,
                                                     'classid_classname.csv'),
                                        sep=',',
                                        header=None,
                                        names=['class_id', 'class_name'],
                                        dtype={
        'class_id': 'UInt16',
        'class_name': 'object'
    })
    args.num_classes = len(classid_classname_dic)

    args.model_name = 'L_16'
    args.interm_features_fc = True
    args.multimodal = False
    args.ret_attn_scores = False
    args.pretrained = False

    main(args)
