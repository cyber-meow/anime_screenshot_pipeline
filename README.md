# Anime Screenshot Dataset Pipeline

Google Colab implementation of this repo

| Notebook Name | Link | Repo |
| --- | --- | --- |
| [Anime Screenshot Pipeline](https://github.com/TheSkinnyRat/anime_screenshot_pipeline_colab/blob/main/anime_screenshot_pipeline.ipynb) | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/TheSkinnyRat/anime_screenshot_pipeline_colab/blob/main/anime_screenshot_pipeline.ipynb) | [![](https://img.shields.io/static/v1?message=Github&logo=github&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://github.com/TheSkinnyRat/anime_screenshot_pipeline_colab) |

---

In this repository I detail the workflow that I have come out to semi-automatically build a dataset from anime for [Stable Diffusion](https://github.com/CompVis/stable-diffusion) fine-tuning.
This is a collection of many resources found on internet (credit to the orignal authors), and some python code written by myself and [ChatGPT](https://chat.openai.com/chat).
I managed to get this done on my personal laptop with 3070 Ti gpu and Ubuntu 22.04 installed.

As for my dataset it is organized in the following way.

**Level 1**
```
├── ./1_character
├── ./2_characters
├── ./3_characters
├── ./4_characters
├── ./5_characters
├── ./6+_charcters
└── ./others
```

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

**Level 3 (the last level)**
```
├── ./1_character
│   ├── ./1_character/AobaKokona
│   │   ├── ./1_character/AobaKokona/face_height_ratio_0-25
│   │   ├── ./1_character/AobaKokona/face_height_ratio_25-50
│   │   ├── ./1_character/AobaKokona/face_height_ratio_50-75
│   │   └── ./1_character/AobaKokona/face_height_ratio_75-100
│   ├── ./1_character/AobaMai
│   │   ├── ./1_character/AobaMai/face_height_ratio_25-50
│   │   ├── ./1_character/AobaMai/face_height_ratio_50-75
│   │   └── ./1_character/AobaMai/face_height_ratio_75-100
...
```

The hierarchical organization aims to auto-balance between different concepts without too much need of worrying about the number of images in each class.
As far as I known, the only trainer that is compatible with such organization now is [EveryDream](https://github.com/victorchall/EveryDream-trainer) (see also [EveryDream2](https://github.com/victorchall/EveryDream2trainer#readme)) with their support of using `multiply.txt`to indicate the number of times that the images of a subfolder should be used during training.
Therefore, I also provide the instruction to generate `multiply.txt` automatically at the end. As for other trainers that are available out there, you will need to modify the data loader yourself for data balance.

On top of this, I also generate a json file for each image to store their metadata. That is, for some image `XXX.png`, there is also a corresponding `XXX.json` of for example the following content
> {"n_faces": 2, "facepos": [[0.20572916666666666, 0.20462962962962963, 0.39791666666666664, 0.5564814814814815], [0.5395833333333333, 0.2657407407407407, 0.7791666666666667, 0.6833333333333333]], "fh_ratio": 0.41759259259259257, "cropped": false, "characters": ["KuraueHinata", "YukimuraAoi"], "general": "aniscreen", "tags": ["looking_at_viewer", "blush", "short_hair", "multiple_girls", "black_hair", "hair_ornament", "2girls", "holding", "twintails", "school_uniform", "green_eyes", "purple_eyes", "collarbone", "upper_body", "grey_hair", "food", "serafuku", "hairclip", "indoors", "holding_food", "onigiri"], "n_people": 2}

Using the json file we can create the caption easily. This will be `XXX.txt` and it may contain some like

> KuraueHinata YukimuraAoi, aniscreen, multiple girls, 2girls, blush, serafuku, school uniform, looking at viewer, food, indoors, onigiri, holding, collarbone, upper body, holding food

Enough for the introduction. Now let me explain in detail how this is achieved! The design of the workflow is quite modular so you may just need some of the following steps.

### Table of Contents

1. [Frame Extraction](#Frame-Extraction)
1. [Similar Image Removal](#Similar-Image-Removal)
1. [Face Detection and Cropping](#Face-Detection-and-Cropping)
1. [Automatic Tagging](#Automatic-Tagging)
1. [Customized Classifier](#Character-Classification-with-Few-Shot-Learning)
1. [Folder Arrangement](#Folder-Arrangement)
1. [Manual Inspection](#Manual-Inspection)
1. [Get Ready for Training](#Get-Ready-for-Training)
1. [Using Fan Arts and Regularization Data](#Using-Fan-Arts-and-Regularization-Data)
1. [Useful Links](#Useful-Links)


### TODO / Potential improvements

- [x] README.md
- [ ] Requirements.txt
- [ ] Create script for removal of similar images as alternative of jupyter notebook
- [x] Separate folder creation, arrangement, and data augmentation from basic metadata generation
- [x] Keep source directory structure in destination directory
- [ ] (Advanced) Combined with existing tagging/captioning tool for manual correction
- [ ] (Advanced) Better few-shot learning model

## Frame Extraction

**Extract 5000~10000 frames per episode of 24 minutes**

**Requirements: `ffmpeg` with [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter and potentially cuda support**

The first thing to do is of course to extract frames from anime. We will use the `ffmpeg` command for this.

```
ffmpeg -hwaccel cuda -i $filename -filter:v \
"mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB" \
-qscale:v 1 -qmin 1 -c:a copy "$prefix"_%d.png
```

The use of [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter removes consecutive frames that are too similar. Enabling the filter makes a big difference in both processing speed and the number of extracted frames (probably 10 times fewer images with the filter).
Therefore, it is important to keep it on.

Of course, typing the command manually for each episode is  cumbersome. Therefore, I make the process easier with the script `extract_frames.py` that runs the command for all the files of a certain pattern from the source directory and save them in separate folder in the destination directory.

```
python extract_frames.py --src_dir /path/to/scr_dir \
--dst_dir /path/to/dst_dir \
--prefix series_episode \
--pattern "my_favorite_anime_*.mp4"
```
With the above command, if `my_favorite_anime_04.mp4` exists in `/path/to/scr_dir`, the aforementioned `ffmppeg` command is run for it and the outputs are saved to `/path/to/dst_dir/EP04/` with names `series_episode_%d.png`.

## Similar Image Removal

**Reduce dataset size by a factor of 10 by removing similar images**

**Requirements: install `fiftyone` with `pip install fiftyone`**

Naturally, there are a lot of similar images in consecutive frames and even with the mpdecimate filter there are still too many similar images. Fortunately, nowadays we can easily find ready-to-use tools that compute image similarity in various ways (for example with neural network representation).

For this repository I use `fiftyone` library and follow the instructions of [this blog post](https://towardsdatascience.com/find-and-remove-duplicate-images-in-your-dataset-3e3ec818b978). The code can be found in `remove_similar.ipynb`. The jupyter notebook allows you to visualize the dataset as shown below

![Screenshot from 2022-12-23 16-02-21](https://user-images.githubusercontent.com/24396911/209356724-a7dd9fea-a46a-40a5-b505-ed4f83895dc3.png)

There are three things to set:
- `dataset_dir` where the images are found (images of all the subdirectories are loaded)
- `thresh` the threshold above which we judge that two images are too similar
- `max_compare_size` the maximum size of a chuck for comparing between images. I am not using the entire dataset here because of memory issue.

Personally I find `thresh=0.985` works well for my example so I do not bother to further visualize the dataset later. I may add a script that allows removing similar images later without launching the jupyter notebook.

#### Remarks

1. If you are also using Ubuntu 22.04 you may encounter problem importing `fiftyone` once it gets installed. I have the problem solved thanks to [this solution](https://github.com/voxel51/fiftyone/issues/1803).
2. Here are some other tools for the same task but I have not tried them
    - https://github.com/ryanfwy/image-similarity
    - https://github.com/ChsHub/SSIM-PIL


## Face Detection and Cropping

**Requirements: [anime-face-detector](https://github.com/hysts/anime-face-detector)**
```
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmpose

pip install anime-face-detector
pip install -U numpy
```

People have been working on anime face detection for years, and now there is an abundance of models out there that do this job right. I am using [hysts/anime-face-detector](https://github.com/hysts/anime-face-detector) since it seems to be more recent and this is also what Kohya S uses in [his blog post](https://note.com/kohya_ss/n/nad3bce9a3622) (in Japanese).
The basic idea here is to detect faces and add their position information to a metadata file and potentially somehow describe them in captions later on. It can also be used for cropping.

### Add face information to metadata

```
python detect_faces.py --src_dir /path/to/src_dir
```

After the above step, a file `XXX.json`is created and contains something like
> {"n_faces": 2, "facepos": [[0.1, 0.4935185185185185, 0.2984375, 0.8481481481481481], [0.5494791666666666, 0.25555555555555554, 0.7578125, 0.6472222222222223]], "fh_ratio": 0.39166666666666666, "cropped": false}

The position format is ``[left, top, right, bottom]``. Since image size may vary relative positions are stored here.

### Crop out the maximum square for each face

```
python crop_faces.py --src_dir /path/to/src_dir --min_face_number 2 
```

This is a simple script to crop the largest possible square images that contain the faces (one for each detected face). The faces would be placed in the middle horizontally and near the top vertically. The corresponding `.json` files are also created. 
The argument `--min_face_number` indicates the minimum number of faces that should be contained in the original image for the cropping to take place.
Note that the cropped images are store in the same folders as the input images and no cropping is performed for face that is too large.



## Automatic Tagging

**Requirements (suggested done in a separate environment)**
```
pip install "tensorflow<2.11"
pip install huggingface-hub
```

### Tag your images with an off-the-shelf tagger

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


## Character Classification with Few-Shot Learning

**Train your own model for series-specific concepts**

**Requirements: [anime-face-detector](https://github.com/hysts/anime-face-detector) as described above and the requirements for [arkel23/animesion](https://github.com/arkel23/animesion)**
```
cd classifier_training
pip install -r requirements.txt
cd models
pip install -e . 
```

General taggers and face detectors are great, but chances are that they do not know anything specific to the anime in question. Therefore, we need to train our own model here. Luckily, foundation models are few-shot learners, so this should not be very difficult _as long as we have a good pretrained model._ Fine tuning the tagger we just used may be a solution. For now, I focus on a even simpler task for good performance, that of **character classification**.

### Basics

I use the great model of [arkel23/animesion](https://github.com/arkel23/animesion). Some weights of anime-related models can be found via this [link](https://drive.google.com/drive/folders/1Tk2e5OoI4mtVBdmjMhfoh2VC-lzW164Q) provided on their [page](https://github.com/arkel23/animesion/tree/main/classification_tagging). I particularly fine tune their best vision only model `danbooruFaces_L_16_image128_batch16_SGDlr0.001_ptTrue_seed0_warmupCosine_interTrue_mmFalse_textLenNone_maskNoneconstanttagtokenizingshufFalse_lastEpoch.ckpt`.

### Dataset Preparation

This is a simple classification task. To begin just create a directory `training_data_original` and organize the directory as following
```
├── ./class1
├── ./class2
├── ./class3
├── ./class4
...
└── ./ood
```

Put a few images in each folder that try to capture different variations of the character.
In my test 10~20 is good enough.
Then, since the classifier is meant to work with head portraits we crop the faces with `classifier_dataset_preparation/crop_and_make_dataset.py`
```
cd classifier_dataset_preparation
python crop_and_make_dataset.py --src_dir /path/to/src_dir --dst_dir /path/to/dst_dir
```

In `dst_dir` you will get the same subdirectories but containing cropped head images instead. I assume the original images only contain one face in each image.
This `dst_dir` will be our classification dataset directory. The next step is to generate `classid_classname.csv` for class id mapping and `labels.csv` for mapping between classes and image paths using `classifier_dataset_preparation/make_data_dic_imagenetsyle.py`
```
python make_data_dic_imagenetsyle /path/to/classification_data_dir
```

Finally if you want to split into training and test set you can run
```
python data_split.py /path/to/classification_data_dir/labels.csv 0.7 0.3
```
The above command does a 70%/30% train/test split.

**Remark.** I find the two scripts `make_data_dic_imagenetsyle.py` and `data_split.py` in the moeImouto dataset folder. Credit to the original author(s) of these two scripts.

### Training

First download `danbooruFaces_L_16_image128_batch16_SGDlr0.001_ptTrue_seed0_warmupCosine_interTrue_mmFalse_textLenNone_maskNoneconstanttagtokenizingshufFalse_lastEpoch.ckpt` from this [link](https://drive.google.com/drive/folders/1Tk2e5OoI4mtVBdmjMhfoh2VC-lzW164Q).
I tried to use the training script provided in [arkel23/animesion](https://github.com/arkel23/animesion) but it does not support customized dataset and transfer learning is somehow broken for [ViT with intermediate feature aggregation](https://www.gwern.net/docs/ai/anime/danbooru/2022-rios.pdf), so I modified the code a little bit and put the modified code in `classifier_training/`. Credit should go to arkel23.
```
python train.py --transfer_learning --model_name L_16 --interm_features_fc \
--batch_size=8 --no_epochs 40 --dataset_path /path/to/classification_data_dir \
--results_dir /path/to/dir_to_save_checkpoint \
--checkpoint_path /path/to/pretrained_checkpoint_dir/danbooruFaces_L_16_image128_batch16_SGDlr0.001_ptTrue_seed0_warmupCosine_interTrue_mmFalse_textLenNone_maskNoneconstanttagtokenizingshufFalse_lastEpoch.ckpt 
```
The three arguments `--transfer_learning`, `--model_name L_16`, and `--interm_features_fc` are crucial while the remaining is to be modified by yourself.
The training fits into the 8GB vram of my poor 3070 Ti Laptop GPU even at batch size 8 and is done fairly quickly due to the small size of the dataset. Note that testing is disabled by default. If you want to use test pass the argument `--use_test_set`. Validation will then be performed every `save_checkpoint_freq` epochs (default to 5).

### Inference

At this point we finally get a model to classify the characters of the series! To use it run
```
python classify_faces.py --dataset_path /path/to/classification_data_dir \
--checkpoint_path ../path/to/checkpoint_dir/trained_checkpoint_name.ckpt \
--src_dir /path/to/image_dir
```

The path of the classification dataset should be provided in order to read the class id mapping csv.
The images of `image_dir` should have their corresponding metadata file containing face position information to be used to create crops fed into the classifier.
We then write the character information into the metadata file if such information is not present (to overwrite in all the cases pass the argument `--overwrite`). At this point, `XXX.json` looks like
> {"n_faces": 2, "facepos": [[0.1, 0.4935185185185185, 0.2984375, 0.8481481481481481], [0.5494791666666666, 0.25555555555555554, 0.7578125, 0.6472222222222223]], "fh_ratio": 0.39166666666666666, "cropped": false, "general": "aniscreen", "tags": ["1girl", "long_hair", "blush", "short_hair", "brown_hair", "hair_ornament", "red_eyes", "1boy", "closed_eyes", "upper_body", "braid", "hairclip", ":o", "parody", "cardigan", "braided_bangs"], "n_people": 2, "characters": ["AobaKokona", "SasaharaYuka"]}

#### Character _unknown_ and _ood_

How to deal with random person from anime that we are not interested in is a delicate question. You can see that I include a class _ood_ (out of distribution) above but this does not seem to be very effective. In consequence, I also set a confidence threshold 0.5 and if the probability of belonging to the classified class is lower than this threshold I set character to _unknown_. Both _unknown_ and _ood_ will be ignored in the following treatments.

## Folder Arrangement

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




## Manual Inspection

At any point of the workflow we can do manual inspection on the generated metadata. For example if you want to tag characters even from behind this can not be achieved with the current workflow as we do not see faces from behind (well I suppose). Using existing guis to modify the metadata files in batch should be a good option.
Otherwise you can simply move things between folders to put them in the right place, and run for example the following script

```
python correct_metadata_from_foldername.py --format '*/character/*' \
--character_list /path/to/character_list --src_dir /path/to/dataset_dir
```

For now it mainly supports correcting character information with folder structure. You should put `*` for any intermediate folder. As it reads from the right putting `character/*` would work as well.
In `character_list` we can provide the list of characters separated by comma to make sure that the folder names are valid.


## Get Ready for Training

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


## Using Fan Arts and Regularization Data

Beyond anime screenshot, you may also want to add fan art and arbitrary regularization images into your dataset. Thanks to the hierarchical structure, we don't need to worry too much about the data imbalance. An example folder structure would be following.

```
├── ./anime_series_1
│   ├── ./screenshots
│   └── .fanart
├── ./anime_series_2
│   ├── ./screenshots
│   └── .fanart
└── ./regularization
    ├── ./regularization_concept_1
    ├── ./regularization_concept_2
    └── ./regularization_concept_3
```
For each of the screenshots folder we then have the structure described previously.

If you only have images for other things you want to add in the dataset, you can repeat the above process. Otherwise, if you use for example [
imgbrd grabber](https://github.com/Bionus/imgbrd-grabber) to download training data, you can also have character, copyright, artist, and tag information about the images.
Personally, I set suffix to `.tag` and text file content to
```
character: %character:spaces,separator=^, %
copyright: %copyright:spaces,separator=^, %
artist: %artist:spaces,separator=^, %
general: %general:spaces,separator=^, %
```
This format is thus also understood by my scripts.

Finally, if you download images and organize them into character directory. You can use `utilities/rename_characters.py` to rename the characters in folder names and in tag files of the above format with the help of a provided csv file ([example](https://github.com/cyber-meow/anime_screenshot_pipeline/blob/main/csv_examplese/character_mapping_example.csv)).
```
python utilities/rename_character.py --src_dir /path/to/src_dir --class_mapping_csv /path/to/character_mapping.csv
```

## Useful Links

### Stable Diffusion

- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [EveryDream1](https://github.com/victorchall/EveryDream-trainer) / [EveryDream2](https://github.com/victorchall/EveryDream2trainer#readme)
- [Linaqruf/kohya-trainer](https://github.com/Linaqruf/kohya-trainer)
- [TheLastBen's fast DreamBooth Colab](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb)
- [Dreambooth Extension for Stable-Diffusion-WebUI](https://github.com/d8ahazard/sd_dreambooth_extension)
- [SD RESOURCE GOLDMINE](https://rentry.org/sdgoldmine)
- [Huggingface Diffusers](https://github.com/huggingface/diffusers) (yeah of course this is beyond Stable Diffusion)

### Some anime models to start with

- [Waifu Diffusion](https://huggingface.co/hakurei/waifu-diffusion-v1-4)
- [ACertainty](https://huggingface.co/JosephusCheung/ACertainty)
- [Anything V-3.0](https://huggingface.co/Linaqruf/anything-v3.0)
- [8528-diffusion](https://huggingface.co/852wa/8528-diffusion)
- [EimisAnimeDiffusion](https://huggingface.co/eimiss/EimisAnimeDiffusion_1.0v)

### Collection of papers about diffusion / score-based generative model
- [heejkoo/Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
- [What's the score?](https://scorebasedgenerativemodeling.github.io/)

### Credits

- [arkel23/animesion](https://github.com/arkel23/animesion)
- [SmilingWolf/wd-v1-4-vit-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger)
- [hysts/anime-face-detector](https://github.com/hysts/anime-face-detector)
- [The blog post about how to remove similar images with `fiftyone`](https://towardsdatascience.com/find-and-remove-duplicate-images-in-your-dataset-3e3ec818b978)
- [Kohya S's training guide](https://note.com/kohya_ss/n/nbf7ce8d80f29#c9d7ee61-5779-4436-b4e6-9053741c46bb)

**And of course, most importantly, all the artists, animators, and staffs that have created so many impressive artworks and animes**, and also the engineers that built Stable Diffusion, the community that curated datasets, and all researchers that work towards the path of better AI technologies
