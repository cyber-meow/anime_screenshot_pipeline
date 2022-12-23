import os
import ast
import random
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms
from transformers import BertTokenizer

from .custom_tokenizer import CustomTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(args, split, label_csv_name='labels.csv'):

    transform = None

    dataset = CustomDataset(args,
                            split=split,
                            label_csv_name=label_csv_name,
                            transform=transform)

    dataset_loader = data.DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.no_cpu_workers,
                                     drop_last=True)

    return dataset, dataset_loader


def get_transform(split, image_size):
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1,
                                   contrast=0.1,
                                   saturation=0.1,
                                   hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform


class CustomDataset(data.Dataset):
    def __init__(self,
                 args,
                 split='train',
                 transform=None,
                 label_csv_name='labels.csv'):
        super().__init__()
        self.root = os.path.abspath(args.dataset_path)
        self.image_size = args.image_size
        self.split = split
        self.transform = transform

        self.tokenizer_method = args.tokenizer
        self.max_text_seq_len = args.max_text_seq_len
        self.shuffle = args.shuffle_tokens
        self.label_csv = os.path.join(self.root, label_csv_name)

        if self.transform is None:
            self.transform = get_transform(split=split,
                                           image_size=self.image_size)

        if self.max_text_seq_len:
            if self.tokenizer_method == 'wp':
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased')
            elif self.tokenizer_method == 'tag':
                self.tokenizer = CustomTokenizer(
                    vocab_path=os.path.join(args.dataset_path, 'labels',
                                            'vocab.pkl'),
                    max_text_seq_len=args.max_text_seq_len)
            self.label_csv = self.set_dir.replace('.csv', '_tags.csv')
            self.df = pd.read_csv(self.label_csv)
        else:
            self.df = pd.read_csv(self.label_csv,
                                  sep=',',
                                  header=None,
                                  names=['class_id', 'dir'],
                                  dtype={
                                      'class_id': 'UInt16',
                                      'dir': 'object'
                                  })

        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.classes = pd.read_csv(os.path.join(self.root,
                                                'classid_classname.csv'),
                                   sep=',',
                                   header=None,
                                   names=['class_id', 'class_name'],
                                   dtype={
                                       'class_id': 'UInt16',
                                       'class_name': 'object'
                                   })
        self.num_classes = len(self.classes)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        img_dir = os.path.join(self.root, 'data', img_dir)
        img = Image.open(img_dir)

        if self.transform:
            img = self.transform(img)

        if self.max_text_seq_len:
            caption = ast.literal_eval(self.df.iloc[idx].tags_cat0)
            if self.shuffle:
                random.shuffle(caption)
            if self.tokenizer_method == 'wp':
                caption = ' '.join(caption)  # originally joined by '[SEP]'
                caption = self.tokenizer(caption,
                                         return_tensors='pt',
                                         padding='max_length',
                                         max_length=self.max_text_seq_len,
                                         truncation=True)['input_ids']
            elif self.tokenizer_method == 'tag':
                caption = self.tokenizer(caption)
            return img, target, caption
        else:
            return img, target

    def __len__(self):
        return len(self.targets)
