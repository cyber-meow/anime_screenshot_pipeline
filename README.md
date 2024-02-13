# Anime2SD

**A 99% automatized pipeline to construct training set from anime and more for text-to-image model training**

Demonstration: [https://youtu.be/-Nzj6SEU9XY?si=8-o9vN6ToTRTeGea](https://youtu.be/-Nzj6SEU9XY?si=8-o9vN6ToTRTeGea)

The old scripts and readme have been moved into [scripts_v1](scripts_v1).

Note that the new naming of metadata follows the convention of [waifuc](https://github.com/deepghs/waifuc) and is thus different from the name given to the older version.
For conversion please use [utilities/convert_metadata.py](utilities/convert_metadata.py).


## Basic Usage

The script `automatic_pipeline.py` allows you to construct a text-to-image training set from anime with minimum effort. All you have to do is

```bash
python automatic_pipeline.py \
    --anime_name name_of_my_favorite_anime \
    --base_config_file configs/pipelines/base.toml \
    --config_file configs/pipelines/screenshots.toml configs/pipelines/booru.toml [...]
```

Providing multiple [configuration files](configs/pipelines) allow for parallel processing of fanarts and animes (and even for parallel processing of multiple animes). You can either create your own configuration files or overwrite existing values by command line arguments.

Of course, you can always go without configuration files if you do not need to run multiple pipelines in parallel.

```bash
python automatic_pipeline.py \
    --start_stage 1 \
    --end_stage 7 \
    --src_dir /path/to/video_dir \
    --dst_dir /path/to/dataset_dir \
    --character_ref_dir /path/to/ref_image_dir \
    --pipeline_type screenshots \
    --crop_with_head \
    --image_prefix my_favorite_anime \
    --ep_init 3 \
    --log_prefix my_favorite_anime
```

:bulb: You can first run from stages 1 to 3 without `--character_ref_dir` to cluster characters. Then you go through the clusters to quickly construct your reference folder and run again from stages 3 to 7 with `--character_ref_dir` now given. See [Wiki](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki) for details.  
:bulb:  Although it is possible to run from stage 0 which downloads anime automatically, it is still recommended to prepare the animes yourself as the downloading part is not fully optimized (may just hang if there are no seeders etc).

There are a lot of arguments (more than 100) that allow you to configure the entire process. See all of them in the aforementioned configuration files or with
```bash
python automatic_pipeline.py --help
```

It is highly recommended to read at least [Main Arguments](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Main-Arguments) so that you know how to set up things correctly.

## Advanced Usage

There are three ways that you can use the script.

- **Use it as a black box:** Type the anime name, go watching several episodes of anime, come back, and the dataset is ready.
- **Use it powerful dataset creation assistant:** You can decide yourself where to start and where to end, with possibility to manually inspect and modify the dataset after each stage and resume. You can provide character reference images, correct character classification results, adjust core tags, edit tags with other tools. This will allow you to construct a better dataset than what we get with the fully automatic process.
- **Use it as a tool box:** Each stage can be run independently for the task in question, with many parameters that you can adjust. Besides the main script, there are also other numerous scripts in this repository that are useful for dataset preparation. However, [waifuc](https://github.com/deepghs/waifuc) which this project heavily makes use of may be more appropriate in this case.

## Pipeline Overview

The script performs all the following automatically.

- [Stage 0] [Anime and fanart downloading](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-0:-Anime-and-Fanart-Downloading)
- [Stage 1] [Frame extraction and similar image removal](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-1:-Frame-Extraction-and-Similar-Image-Removal)
- [Stage 2] [Character detection and cropping ](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-2:-Character-Detection-and-Cropping)
- [Stage 3] [Character classification](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-3:-Character-Classification)
- [Stage 4] [Image selection and resizing](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-4:-Image-Selection-and-Resizing)
- [Stage 5] [Tagging, captioning, and generating wildcards and embedding initialization information](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-5:-Tagging-and-Captioning)
- [Stage 6] [Dataset arrangement](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-6:-Dataset-Arrangement)
- [Stage 7] [Repeat computation for concept balancing](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Stage-7:-Repeat-Computation-for-Concept-Balancing)


## Dataset Organization and Training

- Once we go through the pipeline, the dataset is hierarchically organized in `/path/to/dataset_dir/training` with `multiply.txt` in each subfolder indicating the repeat of the images from this directory. More details on this are provided in [Dataset Organization](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Dataset-Organization).
- Since each trainer reads data differently. Some more steps may be required before training is performed. See [Start Training](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki/Start-Training) for what to do for [EveryDream2](https://github.com/victorchall/EveryDream2trainer), [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), and [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion).

## Installation

1. Clone this directory
    ```bash
    git clone https://github.com/cyber-meow/anime_screenshot_pipeline
    cd anime_screenshot_pipeline
    ```

2.  Depending on your operating system, run either `install.sh` or `install.bat` in terminal
3. Don't forget to activate the environment before running the main script

**Additional Steps and Known Issues**
 
- The first stage of the `screenshots` pipeline uses [ffmpeg](https://ffmpeg.org/) from command line. You can install it with
    - Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    - Windows: `choco install ffmpeg`provided that [Chocolatey](https://chocolatey.org/install) is installed
- If you want to use onnx with gpu, make sure cuda 11.8 is properly installed as [onnxruntime-gpu 1.16 uses cuda 11.8](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) 
- python >= 3.11.0 is not yet supported due to the use of the libtorrent library
- Anime downloading is not working on Windows again due to the use of libtorrent library: https://github.com/arvidn/libtorrent/issues/6689


## Change Logs

### Main

- Update documentation [2023.12.24]
- Fully automatic with only need for specifying anime name [2023.12.02]
- Multi-anime support [2023.12.01]
- Fanart support [2023.12.01]
- .toml support [2023.11.29]
- HCP-diffusion compatibility [2023.10.08]

### Secondary

- Load metadata from text file saved by imgbrd-grabber [2023.12.25]
- Keep tokens separator support for Kohya trainer, possibility to add dropped character tags to the end [2023.12.02]
- Ref directory hierarchy and Character class to account for different appearances of the same character [2023.11.28]
- Embedding initialization with hard tags [2023.11.11]
- Improved classification workflow that takes existing character metadata into account [2023.11.10]
- Core tag-based pruning [2023.10.15]
- Add size to metadata to avoid opening images for size comparison [2023.10.14]


## TODO / Potential improvements

Contributions are welcome

### Secondary

- [ ] Do not crop images that are already cropped before unless otherwise specified
- [ ] Text detection
- [ ] Improve core tag detection by using half body or full body images
- [ ] Bag of words clustering for wildcard
- [ ] Prepare HCP with multiple datasets
- [ ] Arguments to optionally remove subfolders with too few images
- [ ] Replace ffmpeg command by built-in python functions
- [ ] Improved tag pruning (with tag tree?)

### Advanced

- [ ] Beyond character classification: outfits, objects, backgrounds, etc.
- [ ] Image quality filtering 
- [ ] Segmentation and soft mask
- [ ] Graphical interfaces with tagging/captioning tools for manual correction



## Credits

- The new workflow is largely inspired by the fully automatic procedure for single character of [narugo1992](https://github.com/narugo1992) and is largely based on the library [waifuc](https://github.com/deepghs/waifuc)
- The [tag_filtering/overlap_tags.json](tag_filtering/overlap_tags.json) file is provided by gensen2ee
- See the [old readme](scripts_v1/README.md) as well

