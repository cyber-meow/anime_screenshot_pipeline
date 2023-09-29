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

def load_data(args, split):
	
    transform = None

    if args.dataset_name == 'moeImouto':    
        dataset = moeImouto(args, split=split, transform=transform)
    elif args.dataset_name == 'cartoonFace':
        dataset = cartoonFace(root=args.dataset_path,
        image_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'danbooruFaces' or args.dataset_name == 'danbooruFull':
        dataset = danbooruFacesFull(args, split=split, transform=transform)
	
    dataset_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.no_cpu_workers, drop_last=True)

    return dataset, dataset_loader


def get_transform(split, image_size):
	if split == 'train':
		transform = transforms.Compose([
				transforms.Resize((image_size+32, image_size+32)),
				transforms.RandomCrop((image_size, image_size)),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1, 
				contrast=0.1, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5])
				])
	else:
		transform = transforms.Compose([
				transforms.Resize((image_size, image_size)), 
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
				])
	return transform


class danbooruFacesFull(data.Dataset):
	'''
	https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped
	'''
	def __init__(self, args, 
	split='train', transform=None):
		super().__init__()
		self.dataset_name = args.dataset_name
		self.root = os.path.abspath(args.dataset_path)
		self.image_size = args.image_size
		self.split = split
		self.transform = transform

		self.tokenizer_method = args.tokenizer		
		self.max_text_seq_len = args.max_text_seq_len
		self.shuffle = args.shuffle_tokens
		
		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'labels', 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', image_size=self.image_size)

		elif self.split=='val':
			print('Validation set')
			self.set_dir = os.path.join(self.root, 'labels', 'val.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', image_size=self.image_size)

		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'labels', 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', image_size=self.image_size)

		if self.max_text_seq_len:
			if self.tokenizer_method == 'wp':
				self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			elif self.tokenizer_method == 'tag':
				self.tokenizer = CustomTokenizer(
					vocab_path=os.path.join(args.dataset_path, 'labels', 'vocab.pkl'), 
                	max_text_seq_len=args.max_text_seq_len)
			self.set_dir = self.set_dir.replace('.csv', '_tags.csv')
			self.df = pd.read_csv(self.set_dir)
		else:
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
				dtype={'class_id': 'UInt16', 'dir': 'object'})
		
		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'labels', 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		
	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		if self.dataset_name == 'danbooruFaces':
			img_dir = os.path.join(self.root, 'faces', img_dir)
		elif self.dataset_name == 'danbooruFull':
			img_dir = os.path.join(self.root, 'fullMin256', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		if self.max_text_seq_len:
			caption = ast.literal_eval(self.df.iloc[idx].tags_cat0)
			if self.shuffle:
				random.shuffle(caption)
			if self.tokenizer_method == 'wp':
				caption = ' '.join(caption) # originally joined by '[SEP]'
				caption = self.tokenizer(caption, return_tensors='pt', padding='max_length', 
					max_length=self.max_text_seq_len, truncation=True)['input_ids']
			elif self.tokenizer_method == 'tag':
				caption = self.tokenizer(caption)
			return img, target, caption
		else:
			return img, target


	def __len__(self):
		return len(self.targets)


class moeImouto(data.Dataset):
	'''
	https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home
	http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/
	https://github.com/nagadomi/lbpcascade_animeface
	'''
	def __init__(self, args, 
		split='train', transform=None):
		super().__init__()
		self.dataset_name = args.dataset_name
		self.root = os.path.abspath(args.dataset_path)
		self.image_size = args.image_size
		self.split = split
		self.transform = transform

		self.tokenizer_method = args.tokenizer		
		self.max_text_seq_len = args.max_text_seq_len
		self.shuffle = args.shuffle_tokens
		
		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', image_size=self.image_size)			
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', image_size=self.image_size)

		if self.max_text_seq_len:
			if self.tokenizer_method == 'wp':
				self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			elif self.tokenizer_method == 'tag':
				self.tokenizer = CustomTokenizer(
					vocab_path=os.path.join(args.dataset_path, 'labels', 'vocab.pkl'), 
                	max_text_seq_len=args.max_text_seq_len)
			self.set_dir = self.set_dir.replace('.csv', '_tags.csv')
			self.df = pd.read_csv(self.set_dir)
		else:
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
				dtype={'class_id': 'UInt16', 'dir': 'object'})	

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
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
				caption = ' '.join(caption) # originally joined by '[SEP]'
				caption = self.tokenizer(caption, return_tensors='pt', padding='max_length', 
					max_length=self.max_text_seq_len, truncation=True)['input_ids']
			elif self.tokenizer_method == 'tag':
				caption = self.tokenizer(caption)
			return img, target, caption
		else:
			return img, target
		

	def __len__(self):
		return len(self.targets)


class cartoonFace(data.Dataset):
	'''
	http://challenge.ai.iqiyi.com/detail?raceId=5def69ace9fcf68aef76a75d
	https://github.com/luxiangju-PersonAI/iCartoonFace
	'''
	def __init__(self, root, image_size=128, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.image_size = image_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', image_size=self.image_size)
			
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', image_size=self.image_size)

		self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)
