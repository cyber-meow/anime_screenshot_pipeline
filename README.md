# Anime Screenshot Dataset Preparation Pipeline

In this repository I detail the workflow that I have come out to semi-automatically build a dataset from anime for [Stable Diffusion](https://github.com/CompVis/stable-diffusion) fine-tuning.
This is a collection of many resources found on internet (credit to the orignal authors), and some python code written by myself and [ChatGPT](https://chat.openai.com/chat).
I managed to get all the things run in a reasonable amount of time on my personal laptop with 3070 Ti gpu and Ubuntu 22.04 installed (except for training of Stable Diffusion itself, of course).

Working through the workflow, you can get dataset organized in the following way at the end.

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
.
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

The dataset is organized in hierarchy in order to auto-balance between different concepts without too much need of worrying about the number of images in each class.
As far as I known, the only trainer that is compatible with such organization now is [EveryDream](https://github.com/victorchall/EveryDream-trainer) (see also [EveryDream2](https://github.com/victorchall/EveryDream2trainer#readme)) with their support of using `multiply.txt`to indicate the number of times that the images of a subfolder should be used during training.
Therefore, I also provide a script to generate `multiply.txt` automatically. As for other trainers that are available out there, you will need to modify the data loader yourself for data balance.

On top of this, I also generate a json file for each image to store their metadata. That is, for some image `XXX.png`, there is also a corresponding `XXX.json` of for example the following content
> {"count": "2", "characters": ["KuraueHinata", "AobaKokona"], "general": "anishot yamaS2EP04", "facepos": ["fvp 42 70 fhp 22 39", "fvp 15 37 fhp 62 75"], "tags": ["long_hair", "looking_at_viewer", "blush", "short_hair", "open_mouth", "multiple_girls", "skirt", "brown_hair", "shirt", "black_hair", "hair_ornament", "red_eyes", "2girls", "twintails", "purple_eyes", "braid", "pantyhose", "outdoors", "hairclip", "bag", "sunlight", "backpack", "braided_bangs"]}

Using the json file we can create the caption easily. This will be `XXX.txt` and it may contain something like

> 2people, KuraueHinata AobaKokona, anishot yamaS2EP04, fvp 42 70 fhp 22 39 fvp 15 37 fhp 62 75, backpack, outdoors, hair_ornament, pantyhose, braid, short_hair, hairclip, skirt, shirt, long_hair, open_mouth, multiple_girls, brown_hair, bag, twintails

Enough for the introduction. Now let me explain in detail how this is achieved! The design of the workflow is quite modular so you may just need some of the following steps.

### Table of Contents

1. [Frame Extraction](#Frame-Extraction)
1. [Similar Image Removal](#Similar-Image-Removal)
1. [Automatic Tagging](#Automatic-Tagging)
1. [Face Detection](#Face-Detection)
1. [Customized Classifier](#Character-Classification-with-Few-Shot-Learning)
1. [Metadata Generation](#Metadata-Generation)
1. [Folder Rearrangement](#Folder-Arrangement)
1. [Get Ready for Training](#Get-Ready-for-Training)
1. [Using Fan Arts and Regularization Data](#Using-Fan-Arts-and-Regularization-Data)


### TODO / Potential improvements

- [ ] Requirements.txt
- [ ] Create script for removal of similar images as alternative of jupyter notebook
- [ ] Separate folder creation, arrangement, and data augmentation from basic metadata generation
- [ ] Keep source directory structure in destination directory
- [ ] Allow more flexibility in using folder structure for final metadata generation
- [ ] (Advanced) Combined with existing tagging/captioning tool for manual correction
- [ ] (Advanced) Better few-shot learning model

## Frame Extraction

__Extract 5000~10000 frames per episode of 24 minutes__

__Requirements: `ffmpeg` with [mpdecimate](http://underpop.online.fr/f/ffmpeg/help/mpdecimate.htm.gz) filter and potentially cuda support__

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
With the above command, if `my_favorite_anime_04.mp4` exists in `/path/to/scr_dir`, the aforementioned `ffmppeg` command is run for it and the outputs are saved to `/path/to/dst_dir/EP04/` with names `series_episode_%d.png`. I recommend using the `series_episode` pattern without `_` in series and anime names so that latter on we can read this as metadata.

## Similar Image Removal

**Reduce dataset size by a factor of 10 by removing similar images**

**Requirement: install `fiftyone` with `pip install fiftyone`**

Naturally, there are a lot of similar images in consecutive frames and even with the mpdecimate filter there are still too many similar images. Fortunately, nowadays we can easily find ready-to-use tools that compute image similarity in various ways (for example with neural network representation).

For this repository I use `fiftyone` library and follow the instructions of [this blog post](https://towardsdatascience.com/find-and-remove-duplicate-images-in-your-dataset-3e3ec818b978). The code can be found in `remove_similar.ipynb`. The jupyter notebook allows you to visualize the dataset as shown below

![Screenshot from 2022-12-23 16-02-21](https://user-images.githubusercontent.com/24396911/209356724-a7dd9fea-a46a-40a5-b505-ed4f83895dc3.png)

There are three things to set:
- `dataset_dir` where the images are found (images of all the sub-directories are loaded)
- `thresh` the threshold above which we judge that two images are too similar
- `max_compare_size` the maximum size of a chuck for comparing between images. I am not using the entire dataset here because of memory issue.

Personally I find `thresh=0.985` works well for my example so I do not bother to further visualize the dataset later. I may add a script that allows removing similar images later without launching the jupyter notebook.

#### Remarks

1. If you are also using Ubuntu 22.04 you may encounter problem importing `fiftyone` once it gets installed. I have the problem solved thanks to [this solution](https://github.com/voxel51/fiftyone/issues/1803).
2. Here are some other tools for the same task but I have not tried them
    - https://github.com/ryanfwy/image-similarity
    - https://github.com/ChsHub/SSIM-PIL


## Automatic Tagging

**Tag your images with an off-the-shelf tagger**

**Requirements (suggested done in a separate environment):**
```
pip install "tensorflow<2.11"
pip install huggingface-hub
```

Now that we have a preliminary dataset. The next step is to tag the images, again by neural networks. Note this step may be done before or after the face detection step (see pros and cons of each alternative at the end of the next section).

Nowadays we are fortunate enough to have several taggers that work quite well for anime images as we can see from [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger). I simply follow [Kohya S's blog post](https://note.com/kohya_ss/n/nbf7ce8d80f29) (in Japanese) and tag all the images by [SmilingWolf/wd-v1-4-vit-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger/tree/main).

I made the following two simple modifications to Kohya S's script `tag_images_by_wd14_tagger.py`
1. The images of all the sub-directories are tagged
2. To distinguish tags from the actual caption that will be used by the trainer, the associated file for `XXX.png` is `XXX.png.tags` and not `XXX.txt`.

Example usage
```
 python tag_images_by_wd14_tagger.py --batch_size 16 --caption_extension ".tags" /path/to/src_dir
```
Again, credit to Kohya S. for the script and credit to SmilingWolf for the model. Other existing alternative is [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru/releases) which works in a similar way and BLIP caption trained on natural images. Kohya S. also provides corresponding scripts for these. However, I heard that BLIP caption is not very helpful for anime related models.


## Face Detection

#### Order between automatic tagging and face detection

Although these two steps are pretty much independent, there are some pros and cons of using one before another

- Face detection -> Tagging
    I think this order is actually better in theory because we can first remove images with no or too many faces. This is also the only viable direction when we do data augmentation with cropping/padding. However, the way that my code is written now makes the other way around also beneficial.
- Tagging -> Face detection
    The problem is that `detect_faces.py` now does not only detect faces but also arrange folders in sub-directories. In particular it can read tags and determine the number of people in each image and create folder accordingly. This is unfortunate.
    
In the future, I should separate folder creation and data augmentation from face detection. Then face detection, tagging, and customized classifier (next section) can first be run in any order and then we arrange folder according to the generated data.

## Character Classification with Few-Shot Learning

## Metadata Generation

## Folder Arrangement

## Get Ready for Training

## Using Fan Arts and Regularization Data
