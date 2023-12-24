# Dataset Construction Explained

This document details the pipeline of the dataset construction workflow. It is highly recommended to first go through other documents for a more high-level understanding of the script. Note that there two different pipeline types (set with `--pipeline_type`) which could result in slightly different treatments.

#### Table of contents

- [Stage 0] [Anime and Fanart Downloading](#stage-0-anime-and-fanart-downloading)
- [Stage 1] [Frame Extraction and Similar Image Removal](#stage-1-frame-extraction-and-similar-image-removal)
- [Stage 2] [Character Detection and Cropping](#stage-2-character-detection-and-cropping)
- [Stage 3] [Character Classification](#stage-3-character-classification)
- [Stage 4] [Image Selection and Resizing](#stage-4-image-selection-and-resizing)
- [Stage 5] [Tagging and Captioning](#stage-5-tagging-and-captioning)
- [Stage 6] [Dataset Arrangement](#stage-6-dataset-arrangement)
- [Stage 7] [Repeat Computation for Concept Balancing](#stage-7-repeat-computation-for-concept-balancing)


## Stage 0: Anime and Fanart Downloading

**Automatically download animes and images respectively from nyaa.si and Danbooru**

* `--src_dir` is not relevant here
* Output folder for anime: `/path/to/dataset_dir/intermediate/{image_type}/animes`
* Output folder for fanarts: `/path/to/dataset_dir/intermediate/{image_type}/raw`

### Anime downloading

We download anime by searching with the keyword "{submitter name} {anime name} {resolution}", and filter by episode number when possible. Torrent is used for downloading, which means that this stage would hang if there were no seeders. Moreover, the parsing of anime name and episode number is hard coded an may not always work. Therefore it could be simpler for you just to download the animes yourself instead of invoking this stage.

- `anime_name`: Anime name used in the keyword search.  
**Example usage:** --anime_name "yuuki_yuuna_wa_yuusha_de_aru"
- `candidate_submitters`: A list of candidates submitters from which we try to search for anime. Only the first one with which we manage to find an anime to download will be used.  
**Example usage:** --candidate_submitters erai subsplease
- `anime_resolution`: Anime resolution to use in keyword search. Typical choices are 480, 720, and 1080. Defaults to 720.  
**Example usage:** --anime_resolution 1080
- `min_download_episode` and `max_download_episode`: This gives the range of episodes that you want to download. If you want to download all the episodes just leave them as None (the default value).  
**Example usage:** --min_download_episode 2 --max_download_episode 10


### Farart downloading

For now the fanarts are simply downloaded from Danbooru as they come with existing character information. I may add possibility of downloading from other sources later. The downloading can be slow and needs improvement from the waifuc side.

- `anime_name_booru`: Name to search for downloading from booru. It requires it to match exactly the name used on booru. If this is not provided and `--anime_name` is provided, the latter is used.  
**Example usage:** --anime_name_booru "yama_no_susume"
- `character_info_file`: Path to an optional csv file providing correspondence between booru character names and the character names you want to use for training. The character names which are not specified insides remain untouched. Alternatively, you can use it to specify characters that you want to download (in which case you can leave the second column empty). Any characters that are not encountered in the anime downloading phase will then get downloaded if `--download_for_characters` is given.  
**Example usage:** --character_info_file "configs/csv_examples/character_mapping_example.csv"
- `download_for_characters`: Whether to download characters in `--character_info_file` as explained above.  
**Example usage:** --download_for_characters
- `booru_download_limit`: Limit on the total number of images to download from Danbooru. Defaults to no limit. Setting to 0 will download all images as well. Note that if both `--booru_download_limit` and `--booru_download_limit_per_character` are set, we are not guaranteed to download `--booru_download_limit` number of images.  
**Example usage:** --booru_download_limit 1000
- `booru_download_limit_per_character`: Sets a limit on the number of images to download for each character from Danbooru. If set to 0, there will be no limit for each character. The default value is 500.  
**Example usage:** --booru_download_limit_per_character 300
- `allowed_ratings`: Specifies a list of allowed ratings to filter the images downloaded from Danbooru. Options include `s` (safe), `g` (general), `q` (questionable), and `e` (explicit). By default, this list is empty, indicating no filtering based on ratings.  
**Example usage:** --allowed_ratings s g
- `allowed_image_classes`: Defines a list of allowed image classes for filtering the images. Options include `illustration`, `bangumi` (anime), `comic`, and `3d`. By default, only `illustration` and `bangumi` images are downloaded. Set this to an empty list to disable class-based filtering.  
**Example usage:** --allowed_image_classes illustration comic
- `max_download_size`: Sets the maximum size for the smaller dimension of downloaded images. If an image's smaller dimension exceeds this limit, it will be resized. The default value is 1024.  
**Example usage:** --max_download_size 800


## Stage 1: Frame Extraction and Similar Image Removal

**Extract thousands of frames per episode of 24 minutes and remove similar images**

- If you start from this stage, please set `--src_dir` to the folder containing either the videos (for `screenshots` pipelines) or the raw images (for `booru` pipeline).
- Output folder: `/path/to/dataset_dir/intermediate/{image_type}/raw`
- After this stage, you can go over the images to select those you want to keep.

### Frame extraction

**Requirements: `ffmpeg` with [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter; cuda support is optional but highly recommended**

This part is specific to the `screenshots` pipeline. For the moment being, I directly rely on calling `ffmpeg` command like

```
file_pattern="${dst_ep_dir}/${prefix}EP$((ep_init+i))_%d.png"

ffmpeg -hwaccel cuda -i $filename -filter:v \
"mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB" \
-qscale:v 1 -qmin 1 -c:a copy "$file_pattern"
```

The use of [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter removes consecutive frames that are too similar. Enabling the filter makes a big difference in both processing speed and the number of extracted frames (probably 10 times fewer images with the filter). An even more aggressive approach would be to keep only the key frames.

- `extract_key`: Extract key frames instead of using mpdecimate filter.  
**Example usage:** --extract_keys
- `image_prefix`: This allows you to give a prefix to the extracted images. If not provided the prefix corresponding to each video is inferred from file name.  
**Example usage:** --image_prefix haifuri
- `ep_init`: The episode number to start from. Since the processing order is obtained by sorting the filenames, the episode given here could be different from the actual on. If not provided the episode number of each video is inferred from file name.  
**Example usage:** --ep_init 4


:warning: It is important to ensure that every image has different name at this stage. Otherwise some images will be overwritten later.

### Similar image removal

This step uses 'mobilenetv3_large' to detect and remove similar images, reduces dataset size by a factor of 2 to 10 depending on the anime.

Since the extracted images can take a lot of place, I have made the decision to combine the two steps and perform removal for each episode independently after its frames are extracted. A final pass is also performed on all the extracted images at the end to remove repeated images in notably OP and ED.

- `no_remove_similar`: Put this argument to skip this step.  
**Example usage:** --no_remove_similar
- `similar_thresh`: The threshold above which we judge that two images are too similar. Default is 0.95.  
**Example usage:** --similar_thresh 0.9
- `detect_duplicate_model`: You can use any other model from `timm` for duplicate detection.  
**Example usage:** --detect_duplicate_model mobilenetv2_100
- `detect_duplicate_batch_size`: Batch size for computing the features of images that are used for duplicate detection.  
**Example usage:** --detect_duplicate_batch_size 32



## Stage 2: Character Detection and Cropping

**Crop independent characters into a separate folder**

- If you start from this stage, please set `--src_dir` to the folder containing all the images to process (like the `.../raw` folder from the first stage).
- Output folder:  `/path/to/dataset_dir/intermediate/{image_type}/cropped`

This stage along with stage 3 allow to identify the characters of each image. Moreover, the cropped images are also included in the dataset unless `--no_cropped_in_dataset` is specified.

### Command line arguments

- `min_crop_size`: Minimum size for cropped image (shorter edge). Smaller images are dropped. Default is 320.  
  **Example usage:** --min_crop_size 320
- `crop_with_head`: Do not save images without head during cropping.  
  **Example usage:** --crop_with_head
- `crop_with_face`: Do not save images without face during cropping (this can be problematic if you want to learn how to draw characters from behind).  
  **Example usage:** --crop_with_face
- `detect_level`: The level of detection model used. Options are 'n', 's', 'm', 'x'. The 'n' model is faster with less system overhead. For more information please see https://deepghs.github.io/imgutils/main/api_doc/detect/index.html.  
  **Example usage:** --detect_level n
- `use_3stage_crop`: Use 3-stage cropping to additionally get head and half body crops. It can be used at either stage 2 or 4 (if used at stage 4 the cropped images are directly saved to the `classified` folder).  
  **Example usage:** "--use_3stage_crop 2" or "--use_3stage_crop 4"


## Stage 3: Character Classification

**0-shot character clustering or few-shot character classification without training**

- If you start from this stage, please set `--src_dir` to the folder containing images to classify (like the `.../cropped` folder from the second stage or the `.../classified` folder from this stage).
- Output folder:  `/path/to/dataset_dir/intermediate/{image_type}/classified`
- **Recommended:** After this stage, you can go over the images to rename clusters, construct / modify `--character_ref_dir` and rerun from this stage, and move character images around for correction. More details follow.

### Overview

The working horse of this new pipeline is the [ccip embedding](https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html). It allows to perform 0-shot character clustering!

Nonetheless, even if the model knows how to cluster characters, it does not know the names of characters. Moreover, a character can be split into multiple clusters in an undesired way and random people could be wrongly attributed to a certain character class. To overcome this, we rely on both "existing character metadata" and a "character reference directory". Overall, the classification process goes over the following steps:

1. Extract or load CCIP features from the source directory.
2. Perform OPTICS clustering to identify clusters of images.
3. (Optional) Determine labels for images that do not belong to any clusters.
4. Merge clusters based on similarity. This involves:
    - **Mapping clusters to characters using reference images (if provided):** It first utilizes any provided reference images to map clusters directly to specific characters.
    - **Using image metadata to determine character labels for clusters (if available):** If no reference images are provided or after the first step is completed, the process uses available image metadata to assign labels to clusters.
    - **Merging remaining clusters based on similarity:** In the absence of the previous two steps, or when `--keep_unnamed_clusters` is set, any remaining clusters that haven't been labeled through the first two methods are merged based on their similarity. This step ensures that similar characters or images are grouped together even in the absence of reference images or metadata.
5. (Optional) Apply a final filtering step to ensure character consistency.


#### The use of reference directory

Please refer to [Organization of the Character Reference Directory
](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Organization-of-the-Character-Reference-Directory) for instructions on how the reference directory should be organized.

- `character_ref_dir`: Directory containing reference character images.  
  **Example usage:** --character_ref_dir /path/to/character_ref_dir
- `n_add_to_ref_per_character`: Number of additional reference images to add to each character from classification result. Default is 0.  
  **Example usage:** --n_add_to_ref_per_character 20

#### The use of metadata in booru pipeline

Existing character metadata are only taken into account for the `booru` pipeline. In this case, character metadata are not only utilized to determine the labels for clusters, but also to ensure that character assignment is consistently aligned with these metadata. It is important to note that even with this reliance on metadata, the process of character assignment is still essential for the following two reasons:

1. **Ambiguity in Cropped Images:** We are not able to identify the characters of each cropped image based solely on booru metadata when the original image contains multiple characters. 
2. **Enhanced Classification through References:** The use of reference-based classification at this stage allows for a more nuanced and detailed classification. This method can distinguish between different appearances, outfits, and other variations of a character, aspects that are often not detailed in booru metadata.

There are two arguments that are specific to the use of metadata here.

- `ignore_character_metadata`: Ignore existing character metadata during classification (only relevant for `booru` pipeline as metadata are always ignored in `screenshots` pipeline).  
  **Example usage:** --ignore_character_metadata
- `accept_multiple_candidates`: Force label attribution even when multiple candidates are available. This typically happens to a character that always appear with another specific character, or some  form of a character that is recognized as a character tag in booru.   
  **Example usage:** --accept_multiple_candidates


#### Clustering

The initial clustering phase ensures that we include only people who appear with sufficient frequency. The cluster merging phase ensures that we do not end up with too many clusters at the end.

- `cluster_min_samples`: Minimum cluster samples in character clustering. Default is 5.  
  **Example usage:** --cluster_min_samples 10
- `cluster_merge_threshold`: Cluster merge threshold in character clustering. Default is 0.85.  
  **Example usage:** --cluster_merge_threshold 0.75
- `keep_unnamed_clusters`: Keep unnamed clusters when reference images are provided or when characters are available in metadata. Otherwise all images from these clusters are viewed as "noise".   
  **Example usage:** --keep_unnamed_clusters

#### Extract from noise and final filtering

In steps 3 and 5 mentioned in the overview, we may perform extraction to assign unclustered images to existing clusters or labeled classes. Additionally, a final filtering step could be conducted at step 5 to exclude images that significantly differ from others in their respective classes.

- `no_extract_from_noise`: Disable matching character labels for noise images.  
  **Example usage:** --no_extract_from_noise
- `no_filter_characters`: Disable final filtering for character consistency.  
  **Example usage:** --no_filter_characters
- `same_threshold_rel`: Relative threshold for determining whether images belong to the same cluster for noise extraction and filtering. Default is 0.6.  
  **Example usage:** --same_threshold_rel 0.5
- `same_threshold_abs`: Absolute threshold for determining whether images belong to the same cluster for noise extraction and filtering. Default is 20.  
  **Example usage:** --same_threshold_abs 30

Note that for filtering and noise extraction, it is sufficient to for the number or proportion of "similar" images to be larger than the lower of these two thresholds.


### Suggested workflow

As booru images come with metadata information, you may set `--n_add_to_ref_per_character` to a positive number for booru images. In the case of parallel processing, this also ensures that reference-based classification for `screenshots` pipeline only get performed after these reference images are added. Moreover, with `--keep_unnamed_clusters` set and `--end_stage` set to 3, you can further go through the classified sub-folders to refine the reference directory and run this stage again. Finally, you can go through all the resulting sub-folders yourself to check the content and move characters to the correct folder or remove images with no characters, before launching the remaining of the process.

### Remarks

- ccip embeddings are cached.
- You only need to move the images during manual inspection. The script finds the corresponding metadata and cached file themselves as long as they are still under `classified`.
- The character name would be a random string if you run without `--character_ref_folder`. This will be ok for pivotal tuning methods where we associate text embeddings to characters as we can rename embeddings afterwards.
- :warning: Avoid putting comma in character names if you use `--load_aux characters` later as character names are separated by commas and a character name with commas will then be recognized as multiple characters


## Stage 4: Image Selection and Resizing

**Resize images with characters to training folder with resizing**

- If you start from this stage, please set `--src_dir` to the folder containing both `classified` and `raw` (`/path/to/dataset_dir/intermediate/{image_type}` by default).
- Output folder:  `/path/to/dataset_dir/tranining/{image_type}`
- After this stage, you can go over the images to select those you want to keep.

The images obtained after this stage are meant to be the ones used for training.

### Image selection criteria

The folder names from `.../classified` directory are first read and save in the `characters` field of the images' metadata (cropped and original alike). Folder names should of the form `{number}_{chracter_name}` or `{character_name}` as long as it does not start with `{number}_` (otherwise the program parses with the first format). After this, the images are selected in the following way:

- For cropped images: select those with size smaller than half of the original image
- For original images:
    - Selected those with characters
    - Selected `--n_anime_reg` images with no characters for style regularization (use all if there are not enough no character images) in the case of `screenshots` pipeline

:warning: The cropped images are mapped back to the original images using file paths. Thus, moving the original images to another place, removing them, or renaming them would cause errors at this stage. You can use [correct_path_field.py](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/utilities/correct_path_field.py) if the file names remain untouched while the files are moved.

### Command line arguments

- `no_cropped_in_dataset`: Exclude cropped images from the dataset.  
  **Example usage:** --no_cropped_in_dataset
- `no_original_in_dataset`: Exclude original images from the dataset.  
  **Example usage:** --no_original_in_dataset
- `no_resize`: Skip the image resizing process and copy files as they are.  
  **Example usage:** --no_resize
- `max_size`: Maximum image size to resize to, aligning the shorter edge. The default is 768.  
  **Example usage:** --max_size 1024
- `image_save_ext`: Specify the image extension for resized images. The default is `.webp`.  
  **Example usage:** --image_save_ext .jpg
- `filter_again`: Enable filtering of repeated images again at this stage. This is useful since cropped images can be similar even if the full images are not.  
  **Example usage:** --filter_again`
- `overwrite_emb_init_info`: The file `emb_init.json` is also saved in the output folder at this stage. By default the original content of the file is kept if it exists. With this argument the file is completely overwritten.  
  **Example usage:** --overwrite_emb_init_info

#### Arguments specific to screenshots pipeline

- `n_anime_reg`: Set the number of images with no characters to include in the dataset. The default number is 500.  
  **Example usage:** --n_anime_reg 1000

#### Arguments specific to booru pipeline

- `character_overwrite_uncropped`: Overwrite existing character metadata for uncropped images. This is only relevant for the `booru` pipeline as this is always the case otherwise.  
  **Example usage:** --character_overwrite_uncropped
- `character_remove_unclassified`: Remove unclassified characters from the character metadata field.  
  **Example usage:** --character_remove_unclassified


## Stage 5: Tagging and Captioning

**Tag, prune tags, and caption**

- If you start from this stage, please set `--src_dir` to the training folder with images to tag and caption (can be independently used as tagger).
- In-place operation.
- After this stage, you can edit tags and characters yourself using suitable tools, especially if you put `--save_aux processed_tags characters`. More details follow.

### Tagging

In this phase, we use a publicly available taggers to tag images.

- `tagging_method`: Choose a tagger available in [waifuc](https://github.com/deepghs/waifuc).     Choices are 'deepdanbooro', 'wd14_vit', 'wd14_convnext', 'wd14_convnextv2', 'wd14_swinv2', and 'mldanbooru'. Default is 'wd14_convnextv2'.  
  **Example usage:** --tagging_method mldanbooru
- `overwrite_tags`: Overwrite existing tags even if tags exist in metadata.  
  **Example usage:** --overwrite_tags
- `tag_threshold`: Threshold for tagger. Default is 0.35.  
  **Example usage:** --tag_threshold 0.3

### Tag pruning

During fine-tuning, each component presented in the images are more easily associated to the text that better represent it. Therefore, to make sure that the trigger words / embeddings are correctly associated to the key characteristics of the concept, we should remove tags that are inherent to the concept. This can be regarded as a trade-off between "convenience" and "flexibility". Without pruning, we can still pretty much get the concept by including all the relevant tags. However, there is also a risk that these tags get "polluted" and get definitely bound to the target concept.

#### Overview

The pruning process goes over a few steps. You can deactivate tag pruning by setting `--prune_mode none`.

- **Prune blacklisted tags.** Remove tags in the file `--blacklist_tags_file` (one tag per line). Use [blacklist_tags.txt](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/configs/tag_filtering/blacklist_tags.txt) by default.
- **Prune overlap tags.** This includes tags that are sub-string of other tags, and overlapped tags specified in `--overlap_tags_file`. Use [overlap_tags.json](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/configs/tag_filtering/overlap_tags.json) by default.
- **Prune character-related tags.** This stage uses `--character_tags_file`, [character_tags.json](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/configs/tag_filtering/character_tags.json) by default, to prune character-related tags. All the character-related tags below difficulty `--drop_difficulty` are dropped if `--prune_mode` is set to `character`; otherwise for `--prune_mode` set to `character_core` (the default value), only "core tags" of the characters that appear in the images are dropped (see [Core tags](#Core-tags) for details).

The remaining tags are saved to the field `processed_tags` of metadata.

**Character tag difficulty.** We may want to associate a difficulty level to each character tag depending on how hard it is to learn the characteristic. This is done in [character_tags.json](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/configs/tag_filtering/character_tags.json), where we consider two difficulty levels: 0 for human-related tags and 1 for furry, mecha, demon, etc.

#### Common arguments

- `prune_mode`: This argument defines the strategy for tag pruning. The available options are `character`, `character_core`, `minimal`, and `none`, with the default being `character_core`.  
  **Example usage:** --prune_mode character  
  **Description:** Each mode represents a different level of tag pruning:
    - `none`: No pruning is performed, retaining all tags.
    - `minimal`: Only the first two steps mentioned in overview are performed.
    - `character_core`: This mode prunes only the core tags of the relevant characters.
    - `character`: This mode prunes all character-related tags up to `--drop_difficulty`.
- `blacklist_tags_file`, `overlap_tags_file`, and `character_tags_file`: These arguments specify the paths to the files containing different tag filtering information.  
  **Example usage:** --blacklist_tags_file path/to/blacklist_tags.txt
- `process_from_original_tags`: When enabled, the tag processing starts from the original tags instead of previously processed tags.  
  **Example usage:** --process_from_original_tags
- `drop_difficulty`: Determines the difficulty level below which character tags are dropped. Tags with a difficulty level less than this value are added to the drop list, while tags at or above this level are not dropped. The default setting is 2.  
  **Example usage:** --drop_difficulty 1

#### Core tags

A tag is considered a "core tag" of a character if it frequently appears in images containing that character. Identifying these core tags is useful for both deciding which tags should be dropped due to their inherent association with the concept, and determining which tags are suitable for initializing character embeddings in pivotal tuning.

- `compute_core_tag_up_levels`: Specifies the number of directory levels to ascend from the tagged directory for computing core tags. The default is 1, meaning the computation covers all image types.  
  **Example usage:** --compute_core_tag_up_levels 0
- `core_frequency_thresh`: Sets the minimum frequency threshold for a tag to be considered a core tag. The default value is 0.4.  
  **Example usage:** --core_frequency_thresh 0.5
- `use_existing_core_tag_file`: When enabled, uses the existing core tag file instead of recomputing the core tags.  
  **Example usage:** --use_existing_core_tag_file
- `drop_all_core`: If enabled, all core tags are dropped, overriding the `--drop_difficulty` setting.  
  **Example usage:** --drop_all_core
- `emb_min_difficulty`: Sets the minimum difficulty level for tags to be used in embedding initialization. The default is 1.  
  **Example usage:** --emb_min_difficulty 0
- `emb_max_difficulty`: Determines the maximum difficulty level for tags used in embedding initialization. The default is 2.  
  **Example usage:** --emb_max_difficulty 1
- `emb_init_all_core`: If enabled, all core tags are used for embedding initialization, overriding the `--emb_min_difficulty` and `--emb_max_difficulty` settings.  
  **Example usage:** --emb_init_all_core
- `append_dropped_character_tags_wildcard`: Append dropped character tags to the wildcard  
  **Example usage:** --append_dropped_character_tags_wildcard
  
Note that core tags are always computed, and `core_tags.json` and `wildcard.txt` are always saved. However, they are computed at the end of tag processing when `--pruned_mode` is not `character_core`. You can also use [get_core_tags.py](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/utilities/get_core_tags.py) to recompute them.

### Further tag processing

This step consists of tag sorting, optional appending of dropped character tags, and refining the tag list to adhere to the maximum number of tags specified by `--max_tag_number`. Initially, we place certain tags like 'solo', '1girl', '1boy', 'Xgirls', and 'Xboys' at the beginning. Subsequently, the tags are organized based on the `sort_mode`, which determines their order.

- `sort_mode`: Determines the method for sorting tags. Defaults to score.  
  **Example usage:** --sort_mode shuffle  
  **Description:** The available modes offer different approaches to tag ordering:
    - `original`: Maintains the original sequence of the tags as they appear in the data. This mode preserves the initial tag order without any alterations.
    - `shuffle`: Randomizes the order of tags. This mode introduces variety by shuffling the tags into a random sequence, differing for each image.
    - `score`: Sorts tags based on their scores, applicable when using a tagger. In this mode, tags are arranged with those having higher scores placed first, prioritizing the most significant tags.
- `append_dropped_character_tags`: Adds previously dropped character tags back into the tag set, placing them at the end of the tag list.  
  **Example usage:** --append_dropped_character_tags
- `max_tag_number`: Limits the total number of tags included in each image's caption. The default limit is 30.  
  **Example usage:** --max_tag_number 50

### Captioning

This step uses tags and other fields to produce captions, saved both in `caption` field of metadata and as separate `.txt` files.

- `caption_ordering`: Specifies the order in which different information types appear in the caption. The default order is `['npeople', 'character', 'copyright', 'image_type', 'artist', 'rating', 'crop_info', 'tags']`.  
  **Example usage:** --caption_ordering character copyright tags
- `caption_inner_sep`, `caption_outer_sep`, `character_sep`, `character_inner_sep`, `character_outer_sep`: These parameters define the separators for different elements and levels within the captions, such as separating items within a single field or separating different fields  
  **Example usage:** --caption_outer_sep "; "
- `use_[XXX]_prob` arguments: These settings control the probability of including specific types of information in the captions, like character info, copyright info, and others.  
  **Example usage:** --use_character_prob 0.8

For a complete list of all available arguments for captioning, please refer to the configuration file at [configs/pipelines/base.toml](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/configs/pipelines/base.toml).

#### Additional arguments specific to kohya training

- `keep_tokens_sep`: Determines the separator for keep tokens specifically used in Kohya trainer. By default, it uses the value set in `--character_outer_sep`.  
  **Example usage:** --keep_tokens_sep "|| "
- `keep_tokens_before`: Specifies the position in the caption where the `keep_tokens_sep` should be placed before. The default setting is 'tags'.  
  **Example usage:** --keep_tokens_before crop_info

### Manual inspection: Tag and character editing

You can use `--save_aux` to save some metadata fields to separate files and use `--load aux` to load them.

Typically, you can run with `--save_aux processed_tags characters`. You then get files with names like `XXX.processed_tags` and `XXX.characters`. These can be batch edited with tools such as [dataset-tag-editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor) or [BatchPrompter](https://github.com/Snowad14/BatchPrompter). These changes can then be loaded with `--load_aux processed_tags characters`. Remember to run again from this stage to update captions.

- Since tags are not overwritten by default, you don't need to worry about tagger overwriting the edited tags.
- It is better to correct detected characters after stage 3. Here you can edit `XXX.characters` to add non-detected characters.
- :warning: If you edit caption directly, running this stage again will overwrite the changes.


## Stage 6: Dataset Arrangement

**Arrange folder in a certain format for concept balancing**

- If you start from this stage, please set `--src_dir` to the training folder to arrange (`/path/to/dataset_dir/training/{image_type}` by default).
- In-place operation.

For more details please refer to [Dataset Organization](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Dataset-Organization).

### Command line arguments

- `rearrange_up_levels`: This argument specifies the number of directory levels to ascend from the captioned directory when setting the source directory for the rearrange stage. By default, this is set to 0, meaning no change from the captioned directory level.  
  **Example usage:** --rearrange_up_levels 2
- `arrange_format`: It defines the directory hierarchy for dataset arrangement. The default format is `n_characters/character`. Other valid components are `character_string` (useful in the case of further character refinement) and `image_type` (should be used with `--rearrange_up_levels` set to positive values).  
  **Example usage:** --arrange_format n_characters/character_string/image_type
- `max_character_number`:  This argument determines the naming convention for `n_characters` folders. When set, any image containing more than the specified number of characters will be grouped into a single folder named with the format `{n}+_characters`, where n is the number specified. The default value is 6.  
  **Example usage:** `--max_character_number 2`
- `min_images_per_combination`: This sets the minimum number of images required for a specific character combination to have its own directory. If the number of images for a particular character combination is below this threshold, the images are placed in a `character_others` directory. The default number is 10.  
  **Example usage:** `--min_images_per_combination 15`


## Stage 7: Repeat Computation for Concept Balancing

**Balanced dataset by computing repeat/multiply for each sub-folder**

- If you start from this stage, please set `--src_dir` to the training folder containing all training images, screenshots, fanarts, regularization images, or whatever (`/path/to/dataset_dir/training` by default).
- In-place operation.

I assume that we have multiple types of images. They should be all put in the training folder for this stage to be effective.

### Command line arguments

- `compute_multiply_up_levels`: This argument specifies the number of directory levels to ascend from the rearranged directory when setting the source directory for the compute multiply stage. The default value is 1.  
  **Example usage:** --compute_multiply_up_levels 0
- `weight_csv`: This parameter allows the use of a specified CSV file to modify weights during the compute multiply stage.  
  **Example usage:** --weight_csv path/to/custom_weighting.csv
- `min_multiply` and `max_multiply`: These two parameters set respectively the minimum and the maximum repeat for each image. The default values are 1 and 100.  
  **Example usage:** --min_multiply 0.5 --max_multiply 150

### Technical details

We generate here the `multipy.txt` in each image folder to indicate the number of times that the images of this folder should be used in a repeat, with the goal to balance between different concepts during training.

To compute the multiply of each image folder, we first compute its sampling probability. We do this by going through the hierarchy, and at each node, we sample each child with probability proportional to its weight. Its weight is default to 1 but can be changed with the csv file provided through `--weight_csv` ([default_weighting.csv](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/configs/csv_examples/default_weighting.csv) is used by dedfault). It first searches for the folder name of the child directory and next searches for the pattern of the entire path (path of `src_dir` plus path from `src_dir` to the directory) as understood by `fnmatch`.

For example, consider the folder structure
```
├── ./1_character
│   ├── ./1_character/class1
│   └── ./1_character/class2
└── ./others
    ├── ./others/class1
    └── ./others/class3
```
and the csv
```
1_character, 3
class1, 4
*class2, 6
```
For simplicity (and this should be the good practice), assume images are only in the class folders. Then, the sampling probabilities of `./1_character/class1`, `./1_character/class2`, `./others/class1`, and `./others/class3` are respectively 0.75 * 0.4 = 0.3, 0.75 * 0.6 = 0.45, 0.25 * 0.8 = 0.2, and 0.25 * 0.2 = 0.05. Note that the same weight of `class1` can yield different sampling probability because of the other folders at the same level can have different weights (in this case `./1_character/class2` has weight 6 while `./others/class3` has weight 1).

Now that we have the sampling probability of each image folder, we can compute the weight per image by diving it by the number of images in that image folder. Finally, we convert it into multiply by setting the minimum multiply to `--min_multiply` (default to 1). The argument `--max_multiply` sets a hard limit on the maximum multiply of each image folder above which we clip to this value. After running the command you can check the log file to see if you are satisfied with the generated multiplies/repeats.

:warning: The generated multiplies take float values. However, most trainers do not support float repeats. We may thus need to round these values to integers before launching the training process. This is done in both [flatten_folder.py](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/flatten_folder.py) and [prepare_hcp.py](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/prepare_hcp.py).
