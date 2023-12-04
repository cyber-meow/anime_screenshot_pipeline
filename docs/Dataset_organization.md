# Dataset Organization

After the entire process, you will get the following structure in `/path/to/dataset_dir` if you use the default configuration files and run `booru` and `screenshots` pipelines in parallel.

```
.
├── intermediate
│   ├── booru
│   │   ├── classified
│   │   ├── cropped
│   │   └── raw
│   └── screenshots
│       ├── animes
│       ├── classified
│       ├── cropped
│       └── raw
└── training
    ├── booru
    │   ├── 1_character
    │   ├── 2+_characters
    │   └── emb_init.json
    ├── screenshots
    │   ├── 0_characters
    │   ├── 1_character
    │   ├── 2_characters
    │   ├── 3+_characters
    │   └── emb_init.json
    ├── core_tag.json
    ├── emb_init.json
    └── wildcard.txt
```
:bulb:  If `--remove_intermediate` is specified the folders `classified` and `cropped` are removed during the process.

The folder that should be used for training is `/path/to/dataset_dir/training`. Besides the training data, tt contains two important files.
- `emb_init.json` provides information for embedding initialization to be used for pivotal tuning (`emb_init.json` in the subfolders can be ignored).
- `wildcard.txt` provide the wildcard to be used with [sd-dynamic-prompts](https://github.com/adieyal/sd-dynamic-prompts).

You can put other folders, such as your regularization images in the training folder before launching the process so that they will be taken into account as well when we compute the repeat to balance the concept at the end. 

## Organization per Image Type

Each folder `/path/to/dataset_dir/training/{image_type}` is organized in the following way if `--arrange_format` is set to `n_characters/character` (the default value).

**Level 1**
```
├── ./0_characters
├── ./1_character
├── ./2_characters
├── ./3_characters
├── ./4+_characters
```

:bulb: Use `--max_character_number n` so that images containing more than `n` characters are all put together. If you don't want them to be included in the dataset. You can remove it manually.

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
:bulb: Use `--min_images_per_combination m` so that character combinations with fewer than `m` images are all put in the folder `character_others`.  
TODO: Add add an argument to optionally remove them.

The hierarchical organization allows to auto-balance between different concepts without too much need of worrying about the number of images in each class.


## Multi-Anime Dataset and the Like

You can pass the argument `--extra_path_component` to replace  `{image_type}` with `{extra_path_component}/{image_type}` in the aforementioned paths. This allows you for example to have a good organization when processing multiple animes in parallel.

Note that you will need to set `--compute_core_tag_up_levels` to 2 (or even higher number if `--extra_path_component` contains path separators) if you want to have a single wildcard and embedding initialization file for the entire dataset. Similarly, you may want to increase `--rearrange_up_levels` or `--compute_multiply_up_levels` to make sure that dataset balancing is computed from the root training folder.
