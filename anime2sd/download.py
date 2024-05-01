import logging
from typing import List, Dict, Optional, Union

from pynyaasi.nyaasi import NyaaSiClient

from waifuc.action import (
    ModeConvertAction,
    ClassFilterAction,
    # RatingFilterAction,
    AlignMinSizeAction,
)

from .basics import parse_anime_info
from .waifuc_customize import (
    DanbooruSourceWithLimit,
    SaveExporter,
    ConvertSiteMetadataAction,
    RatingFilterActionBooru,
    TagRenameAction,
)


# TODO: Add timeout
def download_animes(
    output_dir: str,
    anime_names: Union[str, List[str]],
    candidate_submitters: Union[str, List[str]],
    resolution: int = 720,
    min_download_episode: Optional[int] = None,
    max_download_episode: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Downloads anime episodes from Nyaa.si based on the provided anime names and
    candidate submitters.

    The function iterates through each anime name and checks for available torrents
    from the specified submitters.
    It downloads the episodes within the given episode number range and saves them
    to the specified output directory.

    Args:
        output_dir (str):
            The directory where the downloaded anime episodes will be saved.
        anime_names (Union[str, List[str]]):
            A single anime name or a list of anime names to download.
        candidate_submitters (Union[str, List[str]]):
            A single submitter name or a list of submitter names to consider.
        resolution (int):
            The resolution of the anime episodes to download. Defaults to 720.
        min_download_episode (Optional[int]):
            The minimum episode number to start downloading from.
            If not provided, downloads from the first available episode.
        max_download_episode (Optional[int]):
            The maximum episode number to download up to.
            If not provided, downloads up to the last available episode.
        logger (Optional[logging.Logger]):
            The logger to use. Defaults to None which uses the default logger.
    """
    # Hide import in function since it does not work on Windows
    from torrentp import TorrentDownloader

    if logger is None:
        logger = logging.getLogger()
    if isinstance(anime_names, str):
        anime_names = [anime_names]
    if isinstance(candidate_submitters, str):
        candidate_submitters = [candidate_submitters]

    client = NyaaSiClient()
    for anime_name in anime_names:
        anime_found = False
        for candidate_submitter in candidate_submitters:
            for item in client.iter_items(
                f"{candidate_submitter} {anime_name} {resolution}"
            ):
                _, ep_num = parse_anime_info(item.title)
                if (
                    min_download_episode
                    and ep_num is not None
                    and ep_num < min_download_episode
                ):
                    continue
                if (
                    max_download_episode
                    and ep_num is not None
                    and ep_num > max_download_episode
                ):
                    continue
                anime_found = True
                torrent_file = TorrentDownloader(item.magnet_url, output_dir)
                logger.info(f"Downloading {item.title} ...")
                torrent_file.start_download()
            if anime_found:
                break


def download_images(
    output_dir: str,
    tags: List[str],
    limit_all: Optional[int] = None,
    limit_per_character: Optional[int] = None,
    ratings: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    max_image_size: Optional[int] = None,
    character_mapping: Optional[Dict[str, str]] = None,
    download_for_characters: bool = True,
    save_aux: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Downloads images from Danbooru based on specified tags and settings,
    saving them to the output directory.
    If `character_mapping` is provided, it renames characters based on the mapping
    and potentially downloads the images for individual characters.

    Args:
        output_dir (str):
            The directory where images will be saved.
        tags (List[str]):
            A list of tags to use for downloading images.
        limit_all (int):
            The limit on the total number of images to download.
        limit_per_character (int):
            The limit on the number of images to download per character.
            Defaults to None which indicates no limit.
        ratings (List[str]):
            A list of ratings for filtering images.
            Should choose between "safe", "r15", and "r18".
        classes (List[str]):
            A list of classes for filtering images.
            Should choose between "illustration", "bangumi", "comic", and "3d".
        max_image_size (int):
            The size for the smaller dimension of the images to resize to when the
            image is too large.
            Defaults to None which indicates no resizing.
        character_mapping (Dict[str, str]):
            A mapping of character tags for renaming.
            Defaults to None which indicates no renaming.
        download_for_characters (bool):
            Flag to indicate whether to download images for individual characters.
            Defaults to False.
        save_aux (List[str]):
            A list of auxiliary data attributes to be saved.
        logger (Logger):
            Logger to use for logging.
    """
    if logger is None:
        logger = logging.getLogger()
    if limit_per_character is not None and limit_per_character <= 0:
        limit_per_character = None

    character_n_images = {}
    if tags:
        logger.info(f"Downloading images for {', '.join(tags)} to {output_dir} ...")
        # This updates the character_n_images dictionary in place
        source = DanbooruSourceWithLimit(
            tags,
            limit_per_character=limit_per_character,
            character_n_images=character_n_images,
        )
        source = source.attach(
            # Convert metadata format
            ConvertSiteMetadataAction(site_name="danbooru"),
            # Preprocess images to RGB and use white background for transparent images
            ModeConvertAction("RGB", "white"),
        )
        if classes:
            # Filtering for classes
            source = source.attach(ClassFilterAction(classes))
        if ratings:
            # Filtering for ratings
            source = source.attach(
                # Disable this as Danbooru source comes with ratings,
                # 'r15' is however not used then
                # RatingFilterAction(ratings),
                RatingFilterActionBooru(ratings),
            )
        if max_image_size:
            # If shorter side is over max_image_size, just resize it to max_image_size
            source = source.attach(AlignMinSizeAction(max_image_size))
        if character_mapping:
            # Renaming characters
            source = source.attach(
                TagRenameAction(character_mapping, fields=["characters"])
            )

        if limit_all is not None and limit_all > 0:
            source = source[:limit_all]
        source.export(
            # Save images (with meta information from danbooru site)
            SaveExporter(output_dir, no_meta=False, save_aux=save_aux)
        )

    if download_for_characters and character_mapping:
        # Download per character images if they are not downloaded yet
        for character in character_mapping.keys():
            if character not in character_n_images:
                download_images(
                    output_dir,
                    tags=[character],
                    limit_all=limit_per_character,
                    limit_per_character=None,
                    max_image_size=max_image_size,
                    character_mapping=character_mapping,
                    download_for_characters=False,
                    save_aux=save_aux,
                    logger=logger,
                )
