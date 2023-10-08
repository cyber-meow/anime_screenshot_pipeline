# Anime Dataset Pipeline

A 99% automatized pipeline to construct training set from anime and more for text-to-image model training

The old scripts and readme have been moved into [scripts_v1](scripts_v1)

Note that the new naming of metadata follows the convention of [waifuc](https://github.com/deepghs/waifuc) and is thus different from the name given to the older version. I may add a script for batch renaming for compatibility later.

**Ensure that you run this script on gpu to have reasonable processing time.**

## Basic Usage

The script `automatic_pipeline.py` allows you to construct a text-to-image training set from anime with minimum effort. All you have to do is

```bash
python automatic_pipeline.py \
    --start_stage 1 \
    --end_stage 7 \
    --src_dir /path/to/video_dir \
    --dst_dir /path/to/dataset_dir \
    --character_ref_dir /path/to/ref_image_dir \
    --image_type screenshots \
    --crop_with_head \
    --image_prefix my_favorite_anime \
    --log_prefix my_favorite_anime
```


The process is split into 7 stages as detailed in [Pipeline Explained](docs/Pipeline.md) / [Wiki](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki). You can decide yourself where to start and where to end, with possibility to manually inspect and modify the dataset after each stage and resume.


- `--src_dir`: The choice of this would vary depending on `start_stage` (details provided in [Pipeline Explained](docs/Pipeline.md) / [Wiki](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki)). In the case where `start_stage` is set to 1, this should be a folder containing the videos to extract frames from.
- `--dst_dir`: Place to construct dataset.
- `--character_ref_dir`: Optional. A folder containing some example images for characters you want to train for. There are two ways to organize
    - With sub-folders: You can put character images in different sub-folders. Sub-folder names are then used as character names.
    - No sub-folders. In this case anything appearing before the first _ in the file name is used as character name.
- `--image_type`: this affects folder names in the constructed dataset (see [Dataset Organization](#Dataset-Organization)) and can also be used in caption (controlled with `--use_image_type_prob`)

:bulb: **Tip:** To filter out characters or random people that you are not interested in, you can use **noise** or any character name that starts with **noise**. This will not be put in the captions later on.  
:bulb: **Tip:** You can first run from stages 1 to 3 without `--character_ref_dir` to cluster characters. Then you go through the clusters to quickly construct your reference folder and run again from stages 3 to 7 with `--character_ref_dir` now given. See [Pipeline Explained](docs/Pipeline.md) / [Wiki](https://github.com/cyber-meow/anime_screenshot_pipeline/wiki) for details.

There are a lot of possible command line arguments that allow you to configure the entire process. See all of them with
```bash
python automatic_pipeline.py --help
```

I may add the possibility to read arguments from `.toml` file later.

## Installation

Clone this directory and install dependencies with
```bash
git clone https://github.com/cyber-meow/anime_screenshot_pipeline
git submodule update --init --recursive

cd anime_screenshot_pipeline

# Use venv, conda or whatever you like here
python -m venv venv
source venv/bin/activate  # Syntax changes according to OS

pip install torch torchvision torchaudio
# add "--index-url https://download.pytorch.org/whl/cu11"
# for windows to install gpu version 
pip install -r requirements.txt
cd waifuc && pip install . && cd ..
# cd waifuc ; pip install . ; cd . for powershell
```

**The first stage of the process also uses [ffmpeg](https://ffmpeg.org/) from command line. Please make sure you can run ffmpege from the command line (ideally with cuda support) for this stage.**

** While I personally work on Linux, others have successfully run the scripts on Windows.

## Dataset Organization

After the entire process, you will get the following structure in `/path/to/dataset_dir` (assume that `image_type` is set to `screenshots`)

```
├── intermediate
│   └── screenshots
│       ├── classified
│       ├── cropped
│       └── raw
└── training
    └── screenshots
```
:bulb: **Tip:** If `--remove_intermediate` is specified the folders `classified` and `cropped` are removed during the process.

The folder that should be used for training is `/path/to/dataset/training`. You can put other folders, such as your regularization images in this folder before launching the process so that they will be taken into account as well when we compute the repeat to balance the concept at the end.

As for `/path/to/dataset/training/sreenshots`, it is organized in th following way

**Level 1**
```
├── ./0_characters
├── ./1_character
├── ./2_characters
├── ./3_characters
├── ./4+_characters
```

:bulb: **Tip:** Use `--max_character_number n` so that images containing more than `n` characters are all put together. If you don't want them to be included in the dataset. You can remove it manually.

**Level 2**
```
├── ./1_character
│   ├── ./1_character/AobaKokona
│   ├── ./1_character/AobaMai
│   ├── ./1_character/KuraueHinata
│   ├── ./1_character/KuraueHinata Hairdown
│   ├── ./1_character/KuraueKenichi
│   ├── ./1_character/KuraueMai
│   ├── ./1_character/KurosakiHonoka
│   ├── ./1_character/KurosakiTaiki
...
```
:bulb: **Tip:** Use `--min_images_per_combination m` so that character combinations with fewer than `m` images are all put in the folder `character_others`.  
TODO: Add add an argument to optionally remove them.

The hierarchical organization allows to auto-balance between different concepts without too much need of worrying about the number of images in each class.
With `multiply.txt` in each folder, this is directly compatible with [EveryDream2](https://github.com/victorchall/EveryDream2trainer). For [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) you need just one more step with `flatten_folder.py`

```
python flatten_folder.py \
    --separator ~ \
    --src_dir /path/to/dataset_dir/training
```

If you do not have the used separator (`~` by default) in any folder name you can undo the change by

```
python flatten_folder.py \
    --separator ~ \
    --src_dir /path/to/dataset_dir/training \
    --revert
```

It is important to switch between the two modes as I rely on the folder structure to compute repeat for now.  
TODO: Compute repeat directly based on metadata and output appropriate format for each trainer (pivotal tuning with [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion) support hopefully coming soon).

## TODO / Potential improvements

Contributions are welcome

### Main

- [x] Readme and Requirements.txt
- [ ] .toml support
- [ ] Fanart support
- [ ] HCP-diffusion compatibility

### Secondary

- [x] Configurable FaceCountAction and HeadCountAction
- [ ] Two-stage classification with small clusters and large clusters
- [ ] Arguments to optionally remove character combinations with too few images
- [ ] Add size to metadata to avoid opening images for size comparison
- [ ] Replace ffmpeg command by built-in python functions
- [ ] Compute repeat based on metadata and trainer-dependent folder organization in the same script
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
