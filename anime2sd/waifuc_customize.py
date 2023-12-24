import os
import re
from typing import List, Dict, Tuple, Optional, Iterator, Union
from PIL import UnidentifiedImageError
from tqdm import tqdm

from waifuc.source.base import BaseDataSource
from waifuc.export.base import LocalDirectoryExporter
from waifuc.action.base import FilterAction, ProcessAction
from waifuc.model import ImageItem
from waifuc.source import WebDataSource, DanbooruSource
from imgutils.detect import detect_faces, detect_heads

from anime2sd.basics import parse_grabber_info


class WebDataSourceWithLimit(WebDataSource):
    """
    Ensure that we do not download more than limit_per_character images per character
    unless there are other characters in the image
    """

    def __init__(
        self,
        *args,
        limit_per_character: Optional[int] = 100,
        character_n_images: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "site_name"), "site_name must be specified"
        self.limit_per_character = limit_per_character
        self.character_n_images = (
            character_n_images if character_n_images is not None else {}
        )

    def _iter_data(self) -> Iterator[Tuple[Union[str, int], str, dict]]:
        for id_, url, meta in super()._iter_data():
            if (
                self.limit_per_character is not None
                and self.site_name in meta
                and "tag_string_character" in meta[self.site_name]
            ):
                characters = re.split(
                    r"\s+", meta[self.site_name]["tag_string_character"]
                )
                # Ensure that we do not download more than limit_per_character
                to_download = False
                for character in characters:
                    if character not in self.character_n_images:
                        self.character_n_images[character] = 0
                    if self.character_n_images[character] < self.limit_per_character:
                        to_download = True
                        break
                if to_download:
                    for character in characters:
                        if character not in self.character_n_images:
                            self.character_n_images[character] = 0
                        self.character_n_images[character] += 1
                    yield id_, url, meta
            else:
                yield id_, url, meta


class DanbooruSourceWithLimit(WebDataSourceWithLimit, DanbooruSource):
    pass


class ConvertSiteMetadataAction(ProcessAction):
    """Retrieve metadata from specific site field"""

    def __init__(
        self,
        site_name: str = "danbooru",
        keep_fields: Optional[List[str]] = None,
    ):
        self.site_name = site_name
        self.keep_fields = keep_fields or ["score", "md5", "rating", "fav_count"]
        self.tag_string_mapping = {
            "tag_string_general": "tags",
            "tag_string_copyright": "copyright",
            "tag_string_artist": "artist",
            "tag_string_character": "characters",
        }

    def process(self, item: ImageItem) -> ImageItem:
        if self.site_name in item.meta:
            for field in self.keep_fields:
                if field in item.meta[self.site_name]:
                    item.meta[field] = item.meta[self.site_name][field]
            for tag_string, field in self.tag_string_mapping.items():
                if tag_string in item.meta[self.site_name]:
                    item.meta[field] = re.split(
                        r"\s+", item.meta[self.site_name][tag_string]
                    )
            item.meta["site"] = self.site_name
            del item.meta[self.site_name]
        return item


class TagRenameAction(ProcessAction):
    """
    Rename tags in metadata according to the provided mapping
    It can also be used to rename other fields that are lists, such as
    characters, copyright, and artist
    """

    def __init__(self, mapping: Dict[str, str], fields: Optional[List[str]] = None):
        # Drop tags that map to empty string or None
        self.mapping = {k: v for k, v in mapping.items() if v}
        self.fields = fields or ["tags", "processed_tags"]
        if isinstance(self.fields, str):
            self.fields = [self.fields]

    def process(self, item: ImageItem) -> ImageItem:
        for field in self.fields:
            if field in item.meta:
                item.meta[field] = [self.mapping.get(t, t) for t in item.meta[field]]
        return item


class RatingFilterActionBooru(FilterAction):
    def __init__(self, ratings: List[str]):
        self.ratings = ratings
        self.rating_mapping = {
            "s": "safe",
            "g": "safe",
            "q": "r18",
            "e": "r18",
        }

    def check(self, item: ImageItem) -> bool:
        rating = item.meta.get("rating", "s")
        return self.rating_mapping[rating] in self.ratings or rating in self.ratings


class MinFaceCountAction(FilterAction):
    def __init__(
        self,
        count: int,
        level: str = "s",
        version: str = "v1.4",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ):
        self.count = count
        self.level = level
        self.version = version
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def check(self, item: ImageItem) -> bool:
        detection = detect_faces(
            item.image,
            self.level,
            self.version,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        return len(detection) >= self.count


class MinHeadCountAction(FilterAction):
    def __init__(
        self,
        count: int,
        level: str = "s",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ):
        self.count = count
        self.level = level
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def check(self, item: ImageItem) -> bool:
        detection = detect_heads(
            item.image,
            self.level,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        return len(detection) >= self.count


class LocalSource(BaseDataSource):
    def __init__(
        self,
        directory: str,
        recursive: bool = True,
        overwrite_path: bool = False,
        load_aux: Optional[List[str]] = None,
        load_grabber_ext: Optional[str] = None,
        progress_bar: bool = True,
    ):
        self.directory = directory
        self.recursive = recursive
        self.overwrite_path = overwrite_path
        self.load_aux = load_aux or []
        self.load_grabber_ext = load_grabber_ext
        self.progress_bar = progress_bar
        self.total_images = self._count_total_images() if progress_bar else None

    def _count_total_images(self):
        return sum(1 for _ in self._iter_files())

    def _iter_files(self):
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        if self.recursive:
            for directory, _, files in os.walk(self.directory):
                group_name = re.sub(r"[\W_]+", "_", directory).strip("_")
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        yield os.path.join(directory, file), group_name
        else:
            group_name = re.sub(r"[\W_]+", "_", self.directory).strip("_")
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

            meta = origin_item.meta
            meta["current_path"] = os.path.abspath(file)
            if "path" not in meta or self.overwrite_path:
                meta["path"] = meta["current_path"]
            if "image_size" not in meta:
                width, height = origin_item.image.size
                meta["image_size"] = [width, height]
            if "filename" not in meta:
                meta["filename"] = os.path.basename(file)
            if "group_id" not in meta:
                meta["group_id"] = group_name

            # Load auxiliary data
            file_basename = os.path.splitext(meta["filename"])[0]
            for attribute in self.load_aux:
                aux_file_path = os.path.join(
                    os.path.dirname(file), file_basename + f".{attribute}"
                )
                if os.path.exists(aux_file_path):
                    with open(aux_file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        items = []
                        for item in content.split(","):
                            item = item.strip()
                            if item != "":
                                items.append(item)
                        meta[attribute] = items

            if self.load_grabber_ext:
                grabber_file_path = file + self.load_grabber_ext
                if os.path.exists(grabber_file_path):
                    with open(grabber_file_path, "r", encoding="utf-8") as f:
                        grabber_info = f.readlines()
                    meta.update(parse_grabber_info(grabber_info))

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
    def __init__(
        self,
        output_dir,
        clear=False,
        in_place=False,
        skip_when_image_exist=True,
        no_meta=False,
        save_caption=True,
        save_aux=None,
    ):
        LocalDirectoryExporter.__init__(self, output_dir, clear)
        self.untitles = 0
        # useful for hierachical structure
        self.in_place = in_place
        self.skip_when_image_exist = skip_when_image_exist
        self.no_meta = no_meta
        self.save_caption = save_caption
        self.save_aux = save_aux

    def export_item(self, item: ImageItem):
        if "filename" in item.meta:
            filename = item.meta["filename"]
        else:
            self.untitles += 1
            filename = f"untited_{self.untitles}.png"

        if self.in_place:
            save_directory = os.path.dirname(item.meta["current_path"])
        else:
            save_directory = self.output_dir
        save_file_path = os.path.join(save_directory, filename)
        item.meta["current_path"] = save_file_path
        # Ideally this is not necessary
        # image size field is modified during action
        width, height = item.image.size
        item.meta["image_size"] = [width, height]
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
        file_basename = os.path.splitext(filename)[0]

        if self.save_caption:
            caption = item.meta.get("caption", None)
            if caption:
                caption_path = os.path.join(save_directory, file_basename + ".txt")
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption)

        if self.save_aux is not None:
            for attribute in self.save_aux:
                content = item.meta.get(attribute, None)
                if content:
                    save_path = os.path.join(
                        save_directory, file_basename + "." + attribute
                    )
                    with open(save_path, "w", encoding="utf-8") as f:
                        if isinstance(content, dict):
                            f.write(", ".join(content.keys()))
                        elif isinstance(content, list):
                            f.write(", ".join(content))
                        elif isinstance(content, str):
                            f.write(content)
                        else:
                            raise ValueError(
                                f"Unsupported exported type {type(content)} "
                                + f"for: {attribute} "
                                + f"of {item.meta['current_path']}"
                            )

        item.save(
            save_file_path,
            no_meta=self.no_meta,
            skip_when_image_exist=self.skip_when_image_exist,
        )

    def reset(self):
        self.untitles = 0
