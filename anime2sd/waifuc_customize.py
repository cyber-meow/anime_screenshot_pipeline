import os
import random
import re
from typing import Iterator
from PIL import UnidentifiedImageError

from waifuc.source.base import RootDataSource
from waifuc.export.base import LocalDirectoryExporter
from waifuc.action.base import ProcessAction
from waifuc.model import ImageItem

from anime2sd import dict_to_caption


class CaptioningAction(ProcessAction):

    def __init__(self, args):
        # TODO: write all the args
        self.args = args

    def process(self, item: ImageItem) -> ImageItem:
        caption = dict_to_caption(item.meta, self.args)
        return ImageItem(item.image, {**item.meta, 'caption': caption})


class LocalSource(RootDataSource):

    def __init__(
            self, directory: str,
            recursive: bool = True, shuffle: bool = False,
            resume_from_subject_and_tags: bool = True):
        self.directory = directory
        self.recursive = recursive
        self.shuffle = shuffle

    def _iter_files(self):
        if self.recursive:
            for directory, _, files in os.walk(self.directory):
                group_name = re.sub(r'[\W_]+', '_', directory).strip('_')
                for file in files:
                    yield os.path.join(directory, file), group_name
        else:
            group_name = re.sub(r'[\W_]+', '_', self.directory).strip('_')
            for file in os.listdir(self.directory):
                yield os.path.join(self.directory, file), group_name

    def _actual_iter_files(self):
        lst = list(self._iter_files())
        if self.shuffle:
            random.shuffle(lst)
        yield from lst

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
            yield ImageItem(origin_item.image, meta)


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
