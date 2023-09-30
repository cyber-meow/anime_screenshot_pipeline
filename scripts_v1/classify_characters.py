import os
import argparse
import json
import csv
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from tensorflow.keras.models import load_model

from vit_animesion import ViT, ViTConfigExtended, PRETRAINED_CONFIGS


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
    for bbox in faces_bbox:
        head_images.append(get_head_image(image, bbox, face_crop_aug))
    return head_images


def get_characters(
        head_images,
        model_cls,
        classid_classname_dic,
        args,
        model_tag,
        tags_all,
        tokenizer,
        device):

    characters = []

    with torch.no_grad():
        if args.multimodal:
            tags_list = get_tags(head_images, model_tag,
                                 tags_all, args.tagger_thresh)
            # print(tags_list)
            captions = torch.vstack([tokenizer(tags) for tags in tags_list])
            # print(captions)
            captions = captions.to(device)
            head_images = prepare_image(head_images, args.image_size, device)
            out_cls = model_cls(head_images, captions)
        else:
            head_images = prepare_image(head_images, args.image_size, device)
            out_cls = model_cls(head_images)
        idxs = torch.argmax(out_cls, dim=1).cpu()
        probs = torch.softmax(out_cls, -1).cpu()
        for idx, prob in zip(idxs, probs):
            idx = idx.item()
            prob = prob[idx]
            if prob > args.cls_thresh:
                class_name = classid_classname_dic.loc[
                    classid_classname_dic['class_id'] == idx,
                    'class_name'].item()
                if class_name == 'ood':
                    class_name = 'unknown'
            else:
                class_name = 'unknown'
            characters.append(class_name)
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


def prepare_image(images, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image_tensors = []
    for image in images:
        image = Image.fromarray(image)
        image_tensors.append(transform(image).unsqueeze(0))

    image_tensors = torch.cat(image_tensors).to(device)
    return image_tensors


def main(args):

    if args.multimodal:
        print('Loading tagger...')

        with open(os.path.join(args.tagger_dir, 'selected_tags.csv'),
                  'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = [row for row in reader]
            header = lines[0]             # tag_id,name,category,count
            rows = lines[1:]
        assert header[0] == 'tag_id' and header[1] == 'name' and header[
            2] == 'category', f"unexpected csv format: {header}"

        tags_all = [row[1] for row in rows[1:] if row[2]
                    == '0']      # categoryが0、つまり通常のタグのみ
        model_tag = load_model(args.tagger_dir)
        tokenizer = CustomTokenizer(
            vocab_path=args.vocabulary_path,
            max_text_seq_len=args.max_text_seq_len)
        args.vocab_size = tokenizer.vocab_size
    else:
        tags_all = model_tag = tokenizer = None
        args.vocab_size = False

    print('Loading classifier...')

    model_cls = VisionTransformer(args)
    state_dict = torch.load(args.checkpoint_path,
                            map_location=torch.device('cpu'))
    model_cls.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cls.to(device)
    model_cls.eval()

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
            print(f'Warning: attribute `characters` found in {json_file}, ' +
                  'skip')
            continue

        head_images = get_head_images(image, metadata, args.face_crop_aug)
        while len(head_images) > 0:
            file_path_batch.append(file_path)
            head_image_batch.append(head_images.pop(0))
            if len(head_image_batch) == args.batch_size:
                characters = get_characters(
                    head_image_batch,
                    model_cls,
                    classid_classname_dic,
                    args,
                    model_tag, tags_all, tokenizer, device)
                for file_path, character in zip(file_path_batch, characters):
                    file_character_dict[file_path].append(character)
                file_path_batch = []
                head_image_batch = []

        if (idx + 1) % args.save_frequency == 0 or idx == len(file_list)-1:
            if len(head_image_batch) > 0:
                characters = get_characters(
                    head_image_batch,
                    model_cls,
                    classid_classname_dic,
                    args,
                    model_tag, tags_all, tokenizer, device)
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


class VisionTransformer(nn.Module):

    def __init__(self, args):
        super(VisionTransformer, self).__init__()

        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.num_classes = args.num_classes
        self.configuration.image_size = args.image_size
        self.configuration.max_text_seq_len = args.max_text_seq_len
        if args.vocab_size:
            self.configuration.vocab_size = args.vocab_size

        load_fc_layer = not args.interm_features_fc
        base_model = ViT(self.configuration,
                         name=args.model_name,
                         pretrained=args.pretrained,
                         load_fc_layer=load_fc_layer,
                         ret_interm_repr=args.interm_features_fc,
                         multimodal=args.multimodal,
                         ret_attn_scores=args.ret_attn_scores)
        self.model = base_model

        if not load_fc_layer:
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

        if hasattr(self, 'text_decoder'):
            predicted_text = self.mlm_head(
                features[:, -self.configuration.max_text_seq_len:, :])
            predicted_text = self.text_decoder(
                predicted_text) + self.decoder_bias

        if hasattr(self, 'text_decoder') and hasattr(self, 'exclusion_loss'):
            return logits, predicted_text, exclusion_loss
        elif hasattr(self, 'text_decoder'):
            return logits, predicted_text
        elif hasattr(self, 'exclusion_loss'):
            return logits, exclusion_loss

        return logits


class CustomTokenizer(object):

    def __init__(self, vocab_path, max_text_seq_len, ret_tensor=True):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            self.vocab_size = len(self.vocab)
        self.max_text_seq_len = max_text_seq_len
        self.ret_tensor = ret_tensor

    def __call__(self, tag_list):
        no_tokens = len(tag_list) + 2
        diff = abs(self.max_text_seq_len - no_tokens)

        tokens = []
        tokens.append(self.vocab('[CLS]'))

        if no_tokens > self.max_text_seq_len:
            tokens.extend([self.vocab(tag)
                          for tag in tag_list[:self.max_text_seq_len-2]])
            tokens.append(self.vocab('[SEP]'))
        elif no_tokens < self.max_text_seq_len:
            tokens.extend([self.vocab(tag) for tag in tag_list])
            tokens.append(self.vocab('[SEP]'))
            tokens.extend([self.vocab('[PAD]') for _ in range(diff)])
        else:
            tokens.extend([self.vocab(tag) for tag in tag_list])
            tokens.append(self.vocab('[SEP]'))

        if self.ret_tensor:
            return torch.tensor([tokens], dtype=torch.int64)
        return tokens

    def decode(self, tokens_list):
        if self.ret_tensor:
            tokens_notensor = tokens_list.squeeze().tolist()
            tag_list = [self.vocab.ret_word(idx) for idx in tokens_notensor]
            return tag_list
        else:
            return [self.vocab.ret_word(idx) for idx in tokens_list]


class Vocabulary(object):
    # https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def ret_word(self, idx):
        if idx not in self.idx2word:
            return '[UNK]'
        return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True,
                        help='Source directory of images')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument(
        '--dataset_path',
        help='Path for the dataset; For classifier id correspondance')
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
        '--multimodal',
        action='store_true',
        help='Use language-vision model instead of vision model')
    parser.add_argument(
        '--face_crop_aug',
        type=float,
        default=1.5,
        help='Ratio between size of the cropped image and that of the face')
    parser.add_argument(
        '--image_size',
        default=128,
        type=int,
        help='Image (square) resolution size')
    parser.add_argument(
        '--max_text_seq_len',
        default=16,
        required=False,
        type=int,
        help='Length for text sequence (for padding and truncation).')
    parser.add_argument(
        '--vocabulary_path',
        type=str, default='classifier_training/vocab.pkl',
        help='path to tag tokenizer')
    parser.add_argument(
        '--cls_thresh',
        type=float,
        default=0.5,
        help="threshold of confidence to classify as character")
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing character metadata')
    parser.add_argument(
        '--tagger_dir',
        type=str, default='tagger/wd14_tagger_model',
        help='directory to store wd14 tagger model')
    parser.add_argument(
        '--tagger_thresh',
        type=float,
        default=0.35,
        help="threshold of confidence to add a tag")
    args = parser.parse_args()

    classid_classname_dic = pd.read_csv(os.path.join(args.dataset_path,
                                                     'classid_classname.csv'),
                                        sep=',',
                                        header=0,
                                        names=['class_id', 'class_name'],
                                        dtype={
        'class_id': 'UInt16',
        'class_name': 'object'
    })
    args.num_classes = len(classid_classname_dic)

    if args.multimodal:
        args.model_name = 'B_16'
        args.interm_features_fc = False
    else:
        args.model_name = 'L_16'
        args.interm_features_fc = True
    args.ret_attn_scores = False
    args.pretrained = False

    main(args)
