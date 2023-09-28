import os
import re
import logging
from typing import Iterator, Optional
from PIL import UnidentifiedImageError
from tqdm import tqdm

from waifuc.source.base import BaseDataSource
from waifuc.export.base import LocalDirectoryExporter
from waifuc.action.base import ProcessAction
from waifuc.model import ImageItem

from anime2sd.captioning import dict_to_caption
from anime2sd.tagging import remove_blacklisted_tags, remove_overlap_tags
from anime2sd.tagging import remove_basic_character_tags, sort_tags


class TagPruningAction(ProcessAction):

    def __init__(self,
                 blacklisted_tags,
                 overlap_tags_dict,
                 pruned_type='character',
                 tags_attribute='processed_tags'):
        assert pruned_type in ['none', 'minimal', 'character']
        self.blacklisted_tags = blacklisted_tags
        self.overlap_tags_dict = overlap_tags_dict
        self.pruned_type = pruned_type
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.pruned_type == 'none':
            return item
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif 'tags' in item.meta:
            tags = item.meta['tags']
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip")
            return item
        tags = remove_blacklisted_tags(tags, self.blacklisted_tags)
        tags = remove_overlap_tags(tags, self.overlap_tags_dict)
        if self.pruned_type == 'character':
            # Only pruned character related tags for character images
            if 'characters' in item.meta and item.meta['characters']:
                tags = remove_basic_character_tags(tags)
        return ImageItem(item.image, {**item.meta, 'processed_tags': tags})


class TagSortingAction(ProcessAction):

    def __init__(self, sort_mode='score',
                 max_tag_number=None,
                 tags_attribute='processed_tags'):
        assert sort_mode in ['original', 'shuffle', 'score']
        self.sort_mode = sort_mode
        self.max_tag_number = max_tag_number
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif 'tags' in item.meta:
            tags = item.meta['tags']
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip")
            return item
        tags = sort_tags(tags, self.sort_mode)
        if self.max_tag_number is not None and len(tags) > self.max_tag_number:
            tags = tags[:self.max_tag_number]
        return ImageItem(item.image, {**item.meta, 'processed_tags': tags})


class TagRemovingUnderscoreAction(ProcessAction):

    def __init__(self, tags_attribute='processed_tags'):
        self.tags_attribute = tags_attribute

    def process(self, item: ImageItem) -> ImageItem:
        if self.tags_attribute in item.meta:
            tags = item.meta[self.tags_attribute]
        # fallback behavior
        elif 'tags' in item.meta:
            tags = item.meta['tags']
        else:
            logging.warning(
                f"{self.tags_attribute} unfound ",
                f"for {item.meta['current_path']}, skip")
            return item
        tags = [self.remove_underscore(tag) for tag in tags]
        result = ImageItem(item.image, {**item.meta, 'processed_tags': tags})
        return result

    @staticmethod
    def remove_underscore(tag):
        if tag == '^_^':
            return tag
        return tag.replace('_', ' ')


class CaptioningAction(ProcessAction):

    def __init__(self, args):
        # TODO: write all the args
        self.args = args

    def process(self, item: ImageItem) -> ImageItem:
        caption = dict_to_caption(item.meta, self.args)
        return ImageItem(item.image, {**item.meta, 'caption': caption})


class LocalSource(BaseDataSource):

    def __init__(self, directory: str,
                 recursive: bool = True,
                 load_aux: Optional[list] = None,
                 progress_bar: bool = True):
        self.directory = directory
        self.recursive = recursive
        self.load_aux = load_aux or []
        self.progress_bar = progress_bar
        self.total_images = (
            self._count_total_images() if progress_bar else None)

    def _count_total_images(self):
        return sum(1 for _ in self._iter_files())

    def _iter_files(self):
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        if self.recursive:
            for directory, _, files in os.walk(self.directory):
                group_name = re.sub(r'[\W_]+', '_', directory).strip('_')
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        yield os.path.join(directory, file), group_name
        else:
            group_name = re.sub(r'[\W_]+', '_', self.directory).strip('_')
            for file in os.listdir(self.directory):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    yield os.path.join(self.directory, file), group_name

    def _iter(self) -> Iterator[ImageItem]:
        for file, group_name in self._iter_files():
            try:
                origin_item = ImageItem.load_from_image(file)
                origin_item.image.load()
            except UnidentifiedImageError:
                continue

            meta = origin_item.meta or {
                'path': os.path.abspath(file),
                'group_id': group_name,
                'filename': os.path.basename(file),
            }
            meta['current_path'] = os.path.abspath(file)

            # Load auxiliary data
            file_basename = os.path.splitext(meta['filename'])[0]
            for attribute in self.load_aux:
                aux_file_path = os.path.join(
                    os.path.dirname(file), file_basename + f'.{attribute}')
                if os.path.exists(aux_file_path):
                    with open(aux_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        items = content.split(',')
                        meta[attribute] = [item.strip() for item in items]

            yield ImageItem(origin_item.image, meta)

    def _iter_from(self) -> Iterator[ImageItem]:
        desc = self.__class__.__name__
        iterator = self._iter()
        if self.progress_bar and self.total_images is not None:
            iterator = tqdm(iterator, total=self.total_images, desc=desc)
        else:
            iterator = tqdm(iterator, desc=desc)
        for item in iterator:
            yield item


class SaveExporter(LocalDirectoryExporter):

    def __init__(self, output_dir,
                 clear=False,
                 in_place=False,
                 skip_when_image_exist=True,
                 no_meta=False,
                 save_caption=True,
                 save_aux=None):

        LocalDirectoryExporter.__init__(self, output_dir, clear)
        self.untitles = 0
        self.in_place = in_place
        self.skip_when_image_exist = skip_when_image_exist
        self.no_meta = no_meta
        self.save_caption = save_caption
        self.save_aux = save_aux

    def export_item(self, item: ImageItem):
        if 'filename' in item.meta:
            filename = item.meta['filename']
        else:
            self.untitles += 1
            filename = f'untited_{self.untitles}.png'

        if self.in_place:
            save_file_path = item.meta['current_path']
        else:
            save_file_path = os.path.join(
                self.output_dir, filename)
        save_directory = os.path.dirname(save_file_path)
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
        file_basename = os.path.splitext(filename)[0]

        if self.save_caption:
            caption = item.meta.get('caption', None)
            if caption:
                caption_path = os.path.join(
                    save_directory, file_basename + '.txt')
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

        if self.save_aux is not None:
            for attribute in self.save_aux:
                content = item.meta.get(attribute, None)
                if content:
                    save_path = os.path.join(
                        save_directory, file_basename + '.' + attribute)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        if isinstance(content, dict):
                            f.write(', '.join(content.keys()))
                        elif isinstance(content, list):
                            f.write(', '.join(content))
                        elif isinstance(content, str):
                            f.write(content)
                        else:
                            raise ValueError(
                                f"Unsupported exported type {type(content)} "
                                + f"for: {attribute} "
                                + f"of {item.meta['current_path']}")

        item.save(save_file_path,
                  no_meta=self.no_meta,
                  skip_when_image_exist=self.skip_when_image_exist)

    def reset(self):
        self.untitles = 0
