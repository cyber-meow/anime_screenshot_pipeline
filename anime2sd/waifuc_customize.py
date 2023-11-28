import os
import re
from typing import Iterator, Optional
from PIL import UnidentifiedImageError
from tqdm import tqdm

from waifuc.source.base import BaseDataSource
from waifuc.export.base import LocalDirectoryExporter
from waifuc.action.base import FilterAction
from waifuc.model import ImageItem
from imgutils.detect import detect_faces, detect_heads


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
        load_aux: Optional[list] = None,
        progress_bar: bool = True,
    ):
        self.directory = directory
        self.recursive = recursive
        self.overwrite_path = overwrite_path
        self.load_aux = load_aux or []
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

            meta = origin_item.meta or {
                "path": os.path.abspath(file),
                "group_id": group_name,
                "filename": os.path.basename(file),
            }
            meta["current_path"] = os.path.abspath(file)
            if "path" not in meta or self.overwrite_path:
                meta["path"] = meta["current_path"]
            if "image_size" not in meta:
                width, height = origin_item.image.size
                meta["image_size"] = [width, height]

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
