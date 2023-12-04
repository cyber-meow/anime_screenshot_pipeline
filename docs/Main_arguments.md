# Main Arguments

While the script has more than 100 arguments. Most of them can be left untouched. For your reference, I am listing here the most important arguments that you probably want to set yourself. Check the configuration files or run with `--help` to see all the arguments.

## Configuration Files

To facilitate the use of the script and to allow for parallel processing, it is possible to save configuration in toml files. There are two arguments for this:

- `base_config_file`: This should be a single file containing the configurations that you want to fix for all your datasets. Normally, you just set it once and forget about it.
- `config_file`: This is a list of files containing extra configurations for the tasks that you want to run. It overwrites the content of `base_config_file`. You would typically set up things like `anime_name`, `character_info_file`, `src_dir`, `character_ref_dir`, and `pipeline_type` in these files. The tasks will then be run in parallel via multithreading.

:bulb: The tasks that are run in parallel may get paused at different stages to wait for other task or skip some stages to avoid repeated treatment, but you do not need to worry about this.

Moreover, the arguments provided directly through command line overwrite those in all the configuration files. Of course, if an argument is provided nowhere, it uses the default value from the script.


## Directories

You probably also want to set up the relevant directories for the construction pipeline.

- `src_dir`: Where to read the source files from. This is ignored if starting from stage 0. See the explanation of each stage to understand how to set it if you start from some specific stage.
- `dst_dir`: Where to construct the dataset. This affects stage 0-4. As for stage 5-7 it follows another logic where we perform in-place operation from either `--src_dir` or a parent folder of it.
- `character_ref_dir`: A directory containing reference images for each character. See [Organization of the Character Reference Directory](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Organization-of-the-Character-Reference-Directory) for details.


## General Pipeline Arguments

Below are some other general arguments that affect globally how the pipeline is run.

- `pipeline_type`: It specifies the pipeline to use; can be set to either `screenshots` or `booru`.
- `image_type`: It affects folder names (see [Dataset Organization](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Dataset-Organization)), and might appear in caption as well. It is treated as an embedding by default. If not provided it is set to `--pipeline_type`.
- `start_stage` and `end_stage`: Where to start and where to end. You can use alias if you want.

**Aliases for different stages**
```
0: "download"  
1: "extract", "remove_similar", or "remove_duplicates"  
2: "crop"  
3: "classify"  
4: "select"  
5: "tag", "caption", or "tag_and_caption"  
6: "arrange"  
7: "balance" or "compute_multiply"  
```


## Logging

Some information gets logged to stdout and the log files along the process.

- `log_dir`: Where to set the log files. Defaults to `logs`. Set to `none` or `None` if you do not want to save log files.
- `log_prefix`: The prefix of the log files. The name of the log file is then `{image_type}_{log_prefix}_{time_information}.txt`. If this is not set it uses `anime_name` if provided and otherwise is set to `logfile`.


## Downloading

Of course, it is important to tell the script what to download.

- `anime_name` and `anime_name_booru`: Names to be used respectively for anime and fanart downloading. If the latter is not set it uses the former by default.
- `character_info_file`: Path to an optional csv file providing correspondence between booru character names and the character names you want to use for training. The character names which are not specified insides remain untouched. Alternatively, you can use it to specify characters that you want to download (in which case you can leave the second column empty). Any characters that are not encountered in anime downloading phase will then get downloaded if `--download_for_characters` is specified (or set to `true` in the configuration files).

:bulb: For `booru` pipeline, you can provide only `character_info_file` with neither `anime_name` nor `anime_name_booru` to download for the characters.


## Other Important Arguments

For the remaining phases the default arguments are mostly good. Still, there are a few arguments that are the most relevant.

- `extract_key`: Set this if you only want to extract key frames in stage 1 of `screenshots` pipeline.
- `use_3stage_crop`: This additionally crops out half body shots and heads of characters, which can be helpful in enhancing the details of the fine-tuned networks. Set it to either to 2 or 4 to indicate at which stage this should be performed. In my provided configuration files this is done for `booru` pipeline at stage 2 but it is not done for `screenshots` pipeline as anime screenshots generally have enough close-up shots.
- `n_add_to_ref_per_character`: How many images to add back to character reference directory after classification phase. This is indispensable for the fully automatic process as it allows fanart images, which come with existing character tags, to be used as reference images to classify screenshot images.
- `keep_unnamed_cluster`: Setting this can be helpful if you want to pause at stage 3 and perform manual inspection, as it allows you to see what exactly are the clusters that are not mapped to known characters. However, if you keep these clusters and perform stage 4, these random strings would appear in final captions and embedding names (well, with pivotal tuning you can always rename the embedding afterwards).
- `overwrite_tags`: Set this if you want to overwrite existing tags (mainly influences whether you use tags from booru or tags from tagger).
