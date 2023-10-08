# Dataset Organization

After the entire process, you will get the following structure in `/path/to/dataset_dir` (assume that `image_type` is set to `screenshots`)

```
├── intermediate
│   └── screenshots
│       ├── classified
│       ├── cropped
│       └── raw
├── training
│   └── screenshots
└── trigger_words.csv
```
:bulb: **Tip:** If `--remove_intermediate` is specified the folders `classified` and `cropped` are removed during the process.

The folder that should be used for training is `/path/to/dataset_dir/training`. You can put other folders, such as your regularization images in this folder before launching the process so that they will be taken into account as well when we compute the repeat to balance the concept at the end.

As for `/path/to/dataset_dir/training/sreenshots`, it is organized in th following way

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
