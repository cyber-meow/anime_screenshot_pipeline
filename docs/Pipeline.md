# Dataset Construction Explained

## Table of Contents

1. [Frame Extraction and Similar Image Removal](#Stage-1:-Frame-Extraction-and-Similar-Image-Removal)
1. [Character Detection and Cropping](#Stage-2:-Character-Detection-and-Cropping)
1. [Character Clustering / Classification](#Stage-3:-Character-Clustering-or-Classification)
1. [Image Selection and Resizing](#Stage-4:-Image-Selection-and-Resizing)
1. [Tagging and Captioning](#Stage-5:-Tagging-and-Captioning)
1. [Folder Arrangement](#Stage-6:-Folder-Arrangement)
1. [Dataset Balancing](#Stage-7:-Dataset-Balancing)

I will also provide below approximate running time on my own laptop with a RTX 3070 Ti.

:bulb: In total it takes me a little more than 2 hours to process four episodes.


## Stage 1: Frame Extraction and Similar Image Removal

**Extract 5000~10000 frames per episode of 24 minutes and remove similar images**

- This stage take 5~10 minutes per episode on my laptop.
- `--src_dir` should be a folder containing the videos to process.
- Output folder: `/path/to/dataset_dir/intermediate/image_type/raw`
- After this stage, you can go over the images yourself to select those you want to keep.

### Frame extraction

**Requirements: `ffmpeg` with [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter and cuda support**

For the moment being, I directly rely on calling `ffmpeg` command like

```
file_pattern="${dst_ep_dir}/${prefix}EP$((ep_init+i))_%d.png"

ffmpeg -hwaccel cuda -i $filename -filter:v \
"mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB" \
-qscale:v 1 -qmin 1 -c:a copy "$file_pattern"
```

**:warning: It is important to ensure that every image has different name at this stage. Otherwise some images will be overwritten later.**

- `--image_prefix`: This allows you to give a prefix to the extracted images.
- `--ep_init`: When you split the process in multiple runs. You should either set different `--image_prefix` or adjust `--ep_init` to specify which episode you start from. Note that the processing order is obtained by sorting the filenames, and thus the episode given here could be different from the actual one if the filename does not have a consistent format.


The use of [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter removes consecutive frames that are too similar. Enabling the filter makes a big difference in both processing speed and the number of extracted frames (probably 10 times fewer images with the filter). For now I have not found a python-only way to achieve the same.  
An even more aggressive approach would be to keep only the key frames (I may add the option later).

### Similar Image Removal

This step uses `fiftyone` and mobilenet to remove similar images, reduces dataset size by a factor of 2 to 10 depending on the anime.

Since the extracted images can take a lot of place, I have made the decision to combine the two steps and perform removal for each episode independently after its frames are extracted. I also do a final pass on all the extracted images at the end to remove repeated images in notably OP and ED.

- `no_remove_similar`: Put this argument to skip this step.
- `similar_thresh`: The threshold above which we judge that two images are too similar. Default is 0.985.
- `detect_duplicate_model`: You can use other model than the default mobilenet. Check `fiftyone` documentation to see what models are available.

:warning: If you are also using Ubuntu 22.04 you may encounter problem importing `fiftyone` once it gets installed. Please check [this thread](https://github.com/voxel51/fiftyone/issues/1803) for some troubleshooting.


## Stage 2: Character Detection and Cropping

**Crop independent characters into a separate folder**

- This stage take 5~10 minutes per episode on my laptop.
- If you start from this stage, please set `--src_dir` to the folder containing all the images to process (like the `.../raw` folder from the first stage).
- Output folder:  `/path/to/dataset_dir/intermediate/image_type/cropped`
- After this stage, you can go over the images yourself to select those you want to keep.

As mentioned, this stage simply crops individual characters out to a new folder using character detection.

- `min_crop_size`: Minimum size for cropped image (shorter edge). Smaller images are dropped. Default is 320.


## Stage 3: Character Clustering or Classification

**0-shot character clustering or few-shot character classification without training**

- This takes ~5 mins for 3k images on my laptop (from 4 episodes)
- If you start from this stage, please set `--src_dir` to the folder containing images to classify (like the `.../cropped` folder from the second stage or the `.../classified` folder from this stage).
- Output folder:  `/path/to/dataset_dir/intermediate/image_type/classified`
- **Recommended:** After this stage, you can go over the images to rename clusters, construct`--character_ref_dir` and rerun from this stage, and move character images around for correction. More details follow.

### Detailed instructions

The working horse of this new pipeline is the [ccip embedding](https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html). It allows to perform 0-shot character clustering!

Nonetheless, even if the model knows how to cluster characters it does not know the names of characters. Moreover, a character can be split into multiple clusters in an undesired way and some character may just contain random people from the anime. There are two ways to fix this.
- Provide `--character_ref_dir` as explained in README to perform few-shot classification.
- Go through the sub-folders of `.../classified`, merge, and rename them accordingly. All the non-relevant characters should go into `-1_noise` folder.

A recommendation is to first run without any reference folder. Use the obtained sub-folders to construct reference folder and run this stage again. Finally, you can go through all the resulting sub-folders your self to check the content and move characters to the correct folder or remove images with no characters.

### Remarks

- ccip embeddings are cached, so you will not compute them again if you start from `.../classified`.
- You only need to move the images during manual inspection. In the next stage the program will find the corresponding metadata and cached file themselves. (For the moment you will not be able resume from this stage without corresponding metadata in the same folder though.)
- The character name would be a random string if you run without `--character_ref_folder`. This will be ok for pivotal tuning methods where we associate text embeddings to characters as we can rename embeddings afterwards.
- :warning: Avoid putting comma in character names if you use `--load_aux characters` later as character names are separated by commas and a character name with commas will then be recognized as multiple characters

### Technical details

TODO

---

TO be updated

## Stage 4: Image Selection and Resizing

- This takes ~1 min per 1k images on my laptop

## Stage 5: Tagging and Captioning

- This takes 1~2 mins per 1k images on my laptop


To supplement our captions we would like also to tag our images.
Nowadays we are fortunate enough to have several taggers that work quite well for anime drawings as we can see from [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger). I simply follow [Kohya S's blog post](https://note.com/kohya_ss/n/nbf7ce8d80f29) (in Japanese) and tag all the images by [SmilingWolf/wd-v1-4-vit-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger/tree/main).

I made the following two simple modifications to Kohya S's script `tag_images_by_wd14_tagger.py`
1. The images of all the subdirectories are tagged
2. To distinguish tags from the actual caption that will be used by the trainer, the associated file for `XXX.png` is `XXX.png.tags` and not `XXX.txt`.

Example usage
```
 python tag_images_by_wd14_tagger.py --batch_size 16 --caption_extension ".tags" /path/to/src_dir
```
Again, credit to Kohya S. for the script and credit to SmilingWolf for the model. Other existing alternative is [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru/releases) which works in a similar way and BLIP caption trained on natural images. Kohya S. also provides corresponding scripts for these. However, I heard that BLIP caption is not very helpful for anime related models.

### Save tag information into metadata

```
python augment_metadata.py --use_tags --general_description "aniscreen" --src_dir /path/to/src_dir
```

This command saves information contained in the tag file into the metadata file. It also computes the number of people in the image with the help of the `[...]girls` and `[...]boys` tags. The text provided by `--general_description` is stored in the attribute `general`.

> {"n_faces": 2, "facepos": [[0.1, 0.4935185185185185, 0.2984375, 0.8481481481481481], [0.5494791666666666, 0.25555555555555554, 0.7578125, 0.6472222222222223]], "fh_ratio": 0.39166666666666666, "cropped": false, "general": "aniscreen", "tags": ["1girl", "long_hair", "blush", "short_hair", "brown_hair", "hair_ornament", "red_eyes", "1boy", "closed_eyes", "upper_body", "braid", "hairclip", ":o", "parody", "cardigan", "braided_bangs"], "n_people": 2}


## Stage 6: Folder Arrangement

- This takes 30 secs to 1 min per 1K images on my laptop

### Only keep images with faces and resize

```
python arrange_folder.py --min_face_number 1 --max_face_number 10 \
--keep_src_structure --format '' --max_image_size 1024 \
--src_dir /path/to/src_dir --dst_dir /path/to/dst_dir
```

The above command can be run before cropping and tagging to eliminate images with no faces.

- With `min_face_number` and `max_face_number` it only saves images whose face number is within this range to `dst_dir`.
- The argument `--keep_src_structure`  makes sure the original folder structure is respected and new folders are only created on top of this structure.
- Passing `--max_image_size` makes sure that saved images are resized so that both its width and height are smaller than the provided value. No resize is perform by default.
- Use `--move_file` if you want to move file to destination directory instead of creating new ones. `max_image_size` is ignored in this case.
    

### Arrange the folder in hierarchy using metadata

For data balance and manual inspection, we can arrange our images into different subfolders.

```
python arrange_folder.py \
--move_file --format 'n_characters/character/fh_ratio' \
--max_character_number 6 --min_image_per_combination 10 \
--src_dir /path/to/src_dir --dst_dir /path/to/dst_dir
```

Using the above command we obtain a folder structure as shown at the beginning. The folder structure itself is specified by the argument `--format`. Different levels of folders are separated by `/`. Accepted folder types are `n_characters`, `n_faces`, `n_people`, `character`, and `fh_ratio`.

- `n_characters`: This creates folders using the number of characters. Passing argument `--max_character_number 6` puts all the scenes with more than 6 characters into the folder `6+_characters`.
- `character`: This creates folders with sorted character names split by `+`. To avoid creating a specific folder for character combination that shows up too few times, we pass the argument `--min_image_per_combination` so that images of all the character combinations with fewer than a certain number of images are saved in `.../character_others`.
- `fh_ratio`: This creates folders according to the maximum face height ratio. The range of percentage of each folder can be changed with `--face_ratio_folder_range` (default to 25).



## Stage 7: Dataset Balancing

We have now an organized dataset and a json file for each image containing its metadata. If you are not going to train locally, you can already upload these data to cloud to be downloaded for further use. You will just need to run the two scripts `generate_captions.py`  and `gnerate_multiply.py` on your training instance (local, colab, runpod, vast.ai, lambdalab, paperspace, or whatever) before launching the trainer.

### Caption generation

This is pretty self-explanatory. It reads the json file and (randomly) generates some caption.
```
python generate_captions.py \
--use_npeople_prob 1 --use_character_prob 1 --use_general_prob 1 \
--use_facepos_prob 0 --use_tags_prob 0.8 --max_tag_numbe 15 --shuffle_tags \
--src_dir /path/to/datset_dir
```

The `use_[...]_prob` arguments specific the probability that a component will be put in the caption on condition that information about this component is stored in the metadata. For face position I used five descriptions (each with two token) for horizontal positions and another five for vertical positions but the model failed to learn its meaning.
We can also pass the arguments `--drop_hair_tag` and/or `--drop_eye_tag` to drop the tags about hair and eyes. This makes sense if we want to teach the model about the concept of specific characters with fixed hair style and eye color.

### Multiply generation

Finally we need to generate the `multipy.txt` in each image folder to indicate the number of times that the images of this folder should be used in a repeat, with the goal to balance between different concepts during training. This is done by
```
python generate_multiply.py --weight_csv /path/to/weight_csv --max_multiply 250 --src_dir /path/to/datset_dir
```

To compute the multiply of each image folder, we first compute its sampling probability. We do this by going through the hierarchy, and at each node, we sample each child with probability proportional to its weight. Its weight is default to 1 but can be changed with a provided csv file ([example](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/csv_examplese/concept_weights_example.csv)). It first searches for the for the folder name of the child directory and next searches for the pattern of the entire path (path of `src_dir` plus path from `src_dir` to the directory) as understood by `fnmatch`.

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
For simplicity (and this should be the good practice) assume images are only in the class folders. Then, the sampling probabilities of `./1_character/class1`, `./1_character/class2`, `./others/class1`, and `./others/class3` are respectively 0.75 * 0.4 = 0.3, 0.75 * 0.6 = 0.45, 0.25 * 0.8 = 0.2, and 0.25 * 0.2 = 0.05. Note that the same weight of `class1` can yield different sampling probability because of the other folders at the same level can have different weights (in this case `./1_character/class2` has weight 6 while `./others/class3` has weight 1).

Now that we have the sampling probability of each image folder, we can compute the weight per image by diving it by the number of images in that image folder. Finally, we convert it into multiply by setting the minimum multiply to 1, and then round the results to integer. The argument `--max_multiply` sets a hard limit on the maximum multiply of each image folder above which we clip to this value. After running the command you can check the log file to see if you are satisfied with the generated multiplies.

### Start fine-tuning

The dataset is ready now. Check [EveryDream](https://github.com/victorchall/EveryDream-trainer) / [EveryDream2](https://github.com/victorchall/EveryDream2trainer#readme) for the fine-tuning of Stable Diffusion.

