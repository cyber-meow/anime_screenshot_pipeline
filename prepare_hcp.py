import argparse
import shutil
import os
import sys
import csv
import glob
import logging
from omegaconf import OmegaConf


def copy_files(src_dir, dst_dir):
    """
    Copy specific files from src_dir to dst_dir.
    """
    files_to_copy = ['train_base.yaml', 'text2img.yaml', 'caption.txt']
    for file in files_to_copy:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))


def get_emb_names_and_inits(filepath):
    """
    Read a CSV file to extract embedding names and their initialization texts.

    Args:
    - filepath: Path to the CSV file.

    Returns:
    - Dictionary with names as keys and initialization texts as values.
    """
    name_init_map = {}

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract the first word from the first column
            name = row[0].split()[0]
            # Extract initialization text or default to '*1'
            init_text = row[1] if row[1] else '*1'
            name_init_map[name] = init_text

    return name_init_map


def modify_main_config_file(filepath, args):
    """
    Modify the main_config_file using OmegaConf
    and save it to the destination directory.
    """

    # Load the original configuration content
    content = OmegaConf.load(filepath)

    # Update the configuration headers
    content.config_dir = os.path.abspath(args.config_dst_dir)
    content.exp_dir_base = os.path.abspath(args.exp_dir)
    content.emb_dir = os.path.abspath(args.emb_dir)
    content.emb_lr = args.emb_lr

    # Update the tokenizer_pt section
    if args.pivotal:
        # Read the embedding names from trigger_word_file
        names = get_emb_names_and_inits(args.trigger_word_file).keys()
        content.tokenizer_pt = {
            'emb_dir': '${emb_dir}',
            'replace': False,
            'train': [{'name': name, 'lr': '${emb_lr}'} for name in names]
        }
    else:
        content.tokenizer_pt = None

    # Save the modified content back to the file
    OmegaConf.save(
        config=content,
        f=os.path.join(args.config_dst_dir, os.path.basename(filepath)))


def modify_dataset_file(filepath, args):
    """
    Modify the dataset.yaml file and save it to the destination directory.
    """

    # Load the original configuration content
    content = OmegaConf.load(filepath)

    # Update the config_dir
    content.config_dir = os.path.abspath(args.config_dst_dir)

    # Get the dataset in question
    dataset = content.data.dataset1

    # Define a template using data_source_1
    template = dataset.source.data_source_1

    # Remove the default data_source_1
    del dataset.source.data_source_1

    # List all subdirectories in dataset_dir
    subdirs = [os.path.join(dp, d)
               for dp, dn, _ in os.walk(args.dataset_dir) for d in dn]

    img_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp')

    data_source_count = 1
    for subdir in subdirs:
        img_files = glob.glob(os.path.join(args.dataset_dir, subdir, '*.*'))
        # Check if the directory contains any images
        if any(file.lower().endswith(img_exts) for file in img_files):
            # Update fields using the template
            new_data_source = OmegaConf.create(template)
            new_data_source.img_root = os.path.join(args.dataset_dir, subdir)

            # Update repeat based on multiply.txt
            multiply_file = os.path.join(
                args.dataset_dir, subdir, 'multiply.txt')
            if os.path.exists(multiply_file):
                with open(multiply_file, 'r') as file:
                    repeat_val = round(float(file.readline().strip()))
                new_data_source.repeat = repeat_val
            else:
                logging.warning(
                    f"Directory {subdir} does not have multiply.txt. "
                    "Setting repeat to 1.")
                new_data_source.repeat = 1

            # Update caption_file path
            new_data_source.caption_file.path = os.path.join(
                args.dataset_dir, subdir)

            # Add the new data source to the content
            data_source_name = f"data_source_{data_source_count}"
            dataset.source[data_source_name] = new_data_source
            data_source_count += 1

    # Save the modified content back to the file
    OmegaConf.save(config=content, f=os.path.join(
        args.config_dst_dir, os.path.basename(filepath)))


def create_embeddings(args):
    """
    Create embeddings and save them to emb_dir.
    """
    from hcpdiff.tools.create_embedding import PTCreator

    content = OmegaConf.load(args.main_config_file)
    pretrained_model = content.model.pretrained_model_name_or_path
    pt_creator = PTCreator(pretrained_model, args.emb_dir)
    # Read the embedding names from trigger_word_file
    names_inits = get_emb_names_and_inits(args.trigger_word_file)
    for name in names_inits:
        pt_creator.creat_word_pt(
            name, args.emb_n_words, names_inits[name], replace=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prepare HCP configurations and embeddings.")

    parser.add_argument('--config_src_dir',
                        default='hcp_configs',
                        help='Source directory for config files.')
    parser.add_argument('--config_dst_dir',
                        required=True,
                        help='Destination directory for config files.')
    parser.add_argument('--dataset_dir',
                        required=True,
                        help='Directory containing the dataset.')
    parser.add_argument('--exp_dir',
                        default=None,
                        help=('Experiment directory. '
                              'Default is {config_dst_dir}/exps.'))

    parser.add_argument('--main_config_file',
                        default='hcp_configs/lora_conventional.yaml',
                        help='Path to the main configuration file.')

    parser.add_argument('--pivotal',
                        action='store_true',
                        help='Flag to indicate pivotal tuning.')
    parser.add_argument('--trigger_word_file',
                        required='--pivotal' in sys.argv,
                        help=('File with trigger words for embeddings. '
                              'Required for pivotal tuning.'))
    parser.add_argument('--emb_dir',
                        default=None,
                        help=('Directory for saving embeddings. '
                              'Default is {config_dst_dir}/embs.'))
    parser.add_argument('--emb_n_words',
                        default=2,
                        type=int,
                        help=('Number of tokens for each embedding.'))
    parser.add_argument('--emb_lr',
                        default=1e-3,
                        type=float,
                        help='Learning rate for embeddings. Default is 1e-3.')

    args = parser.parse_args()
    
    if args.exp_dir is None:
        args.exp_dir = os.path.join(args.config_dst_dir, 'exps')

    if args.emb_dir is None:
        args.emb_dir = os.path.join(args.config_dst_dir, 'embs')

    os.makedirs(args.config_dst_dir, exist_ok=True)
    # Copy files
    copy_files(args.config_src_dir, args.config_dst_dir)

    # Modify files
    modify_main_config_file(args.main_config_file, args)
    modify_dataset_file(
        os.path.join(args.config_src_dir, 'dataset.yaml'), args)

    # Create embeddings if pivotal flag is provided
    if args.pivotal:
        create_embeddings(args)
