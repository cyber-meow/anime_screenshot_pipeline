# Start Training

Once we go through the pipeline, the dataset is hierarchically organized in `/path/to/dataset_dir/training` with `multiply.txt` in each subfolder indicating the repeat of the images from this directory. You can pretty much launch the training process with your favorite trainer at this stage, modulo a few more steps to make sure that the data are read correctly.


## Training with EveryDream

With `multiply.txt` in each folder, the above structure is directly compatible with [EveryDream2](https://github.com/victorchall/EveryDream2trainer). 

## Training with kohya trainer

For [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) you need to perform one more step with `flatten_folder.py`

```bash
python flatten_folder.py \
    --separator ~ \
    --src_dir /path/to/dataset_dir/training
```

If you do not have the used separator (`~` by default) in any folder name you can undo the change by

```bash
python flatten_folder.py \
    --separator ~ \
    --src_dir /path/to/dataset_dir/training \
    --revert
```

It is important to switch between the two modes as I rely on the folder structure to compute repeat for now.

## Training with HCP-Diffusion

[HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion) requires to set up an yaml file to specify the repeat of each data source, and its configuration is generally more complicated, so I have provided `prepare_hcp.py` to streamline the process (to be run in the hcp-diffusion python environment).

```bash
python prepare_hcp \
    --config_dst_dir /path/to/training_config_dir \
    --dataset_dir /path/to/dataset_dir/training
    --pivotal \
    --trigger_word_file /path/to/dataset_dir/emb_init.json
```

Once this is done, the embeddings are created in `/path/to/training_config_dir/embs` and you can start training with

```bash
accelerate launch -m hcpdiff.train_ac_single \
    --cfg /path/to/training_config_dir/lora_conventional.yaml
```

### Further details
- `--pivotal` indicates pivotal tuning, i.e. training of embedding and network at the same time (this is not possible with neither kohya nor EveryDream). Remove this argument if you do not want to train embedding.
- You can customize the embedding you want to create and how they are initialized by modifying the content of `emb_init.json`.
- Use `--help` to see more arguments. Notably you can set `--emb_dir`, `--exp_dir`, and `--main_config_file` (which defaults to `hcp_configs/lora_conventional.yaml`), among others.
- To modify training and dataset parameters, you can modify either directly the files in `hcp_configs` before running the script or modify `dataset.yaml` and `lora_conventional.yaml` (or other config file you use) in `/path/to/training_config_dir` after running the script.
- You should not move the generated config files because some absolute paths are used.


## Training with ...

Each trainer has its strength and drawback. If you know another good trainer that I overlook here, please let me know.
