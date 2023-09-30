import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import wandb
import torch


def ret_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path for the dataset.')
    parser.add_argument('--dataset_name',
                        help='Name of the dataset.',
                        default='anime')
    parser.add_argument('--model_name',
                        choices=[
                            'shallow', 'resnet18', 'resnet50', 'resnet152',
                            'efficientnetb0', 'B_16', 'B_32', 'L_16', 'L_32'
                        ],
                        default='B_16',
                        help='Which model architecture to use')
    parser.add_argument('--results_dir',
                        default='results_training',
                        type=str,
                        help='The directory where results will be stored')
    parser.add_argument('--image_size',
                        default=128,
                        type=int,
                        help='Image (square) resolution size')
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='Batch size for train/val/test.')
    parser.add_argument('--no_epochs',
                        default=50,
                        type=int,
                        help='Total number of epochs for training.')
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_scheduler',
                        type=str,
                        choices=['warmupCosine', 'epochDecayConstant'],
                        default='warmupCosine',
                        help='LR scheduler.')
    parser.add_argument(
        '--epoch_decay',
        default=20,
        type=int,
        help='After how many epochs to decay the learning rate once.')
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=1000,
                        help='Warmup steps for LR scheduler.')
    parser.add_argument('--pretrained',
                        action='store_true',
                        help='For models with pretrained weights available'
                        'Default=False')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--transfer_learning',
                        action='store_true',
                        help='Load partial state dict for transfer learning'
                        'Resets the [embeddings, logits and] fc layer for ViT'
                        'Resets the fc layer for Resnets'
                        'Default=False')
    parser.add_argument(
        '--load_partial_mode',
        choices=[
            'full_tokenizer', 'patchprojection', 'posembeddings', 'clstoken',
            'patchandposembeddings', 'patchandclstoken',
            'posembeddingsandclstoken', None
        ],
        default=None,
        help='Load pre-processing components to speed up training')
    parser.add_argument(
        '--log_freq',
        default=10,
        type=int,
        help='Frequency in steps to print results (and save images if needed).'
    )
    parser.add_argument('--save_checkpoint_freq',
                        default=5,
                        type=int,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--no_cpu_workers',
                        type=int,
                        default=4,
                        help='CPU workers for data loading.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed for initialization')
    parser.add_argument(
        '--interm_features_fc',
        action='store_true',
        help=
        'If use this flag creates FC using intermediate features instead of only last layer.'
    )
    parser.add_argument(
        '--debugging',
        action='store_true',
        help=
        'If use this flag then shortens the training/val loops to log_freq*3.')
    parser.add_argument('--exclusion_loss',
                        action='store_true',
                        help='Use layer-wise exclusion loss')
    parser.add_argument('--temperature',
                        type=float,
                        default=1.0,
                        help='Temperature for exclusion loss')
    parser.add_argument('--exclusion_weight',
                        type=float,
                        default=0.01,
                        help='Weight for exclusion loss')
    parser.add_argument(
        '--exc_layers_dist',
        type=int,
        default=2,
        help='Number of layers in between to calculate exclusion')
    parser.add_argument('--multimodal',
                        action='store_true',
                        help='Vision+tags if true')
    parser.add_argument(
        '--max_text_seq_len',
        default=None,
        required=False,
        type=int,
        help=
        'Length for text sequence (for padding and truncation). Default 16.')
    parser.add_argument('--mask_schedule',
                        choices=[None, 'bert', 'full', 'sigmoid'],
                        default=None,
                        help='Scheduler for masking language tokens.')
    parser.add_argument('--mask_wu_percent',
                        type=float,
                        default=0.1,
                        help='Percentage of training steps for masks warmup')
    parser.add_argument('--mask_cd_percent',
                        type=float,
                        default=0.5,
                        help='Percentage of training steps for masks cooldown')
    parser.add_argument('--ret_attn_scores',
                        action='store_true',
                        help='Returns attention scores for visualization')
    parser.add_argument(
        '--tokenizer',
        type=str,
        choices=['wp', 'tag'],
        default='tag',
        help=
        'Tokenize using word-piece (BERT pretrained from HF) or custom tag-level'
    )
    parser.add_argument(
        '--masking_behavior',
        type=str,
        choices=['constant', 'random'],
        default='constant',
        help='When masked convert token to 1 or to a random int in vocab size')
    parser.add_argument(
        '--shuffle_tokens',
        action='store_true',
        help=
        'When turned on it shuffles tokens before sending to bert or custom tokenizer'
    )
    return parser


def print_write(f, line):
    f.write('{}\n'.format(line))
    print(line)


def decode_text(f,
                tokenizer,
                outputs_text,
                captions,
                captions_updated,
                labels_text,
                num_print=1,
                save_all_captions=False):
    _, text_pred = torch.topk(outputs_text,
                              1,
                              dim=2,
                              largest=True,
                              sorted=True)
    text_pred = text_pred.squeeze()
    for j, sample in enumerate(text_pred):
        if j >= num_print:
            break
        else:
            if not save_all_captions:
                curr_line = '''Prediction: {}
                Ground truth: {}
                Input tokens: {}
                Ground truth tokens: {}
                Labels (masks): {}'''.format(
                    tokenizer.decode(sample),
                    tokenizer.decode(captions[j].data),
                    captions_updated[j].data, captions[j].data,
                    labels_text[j].data)
                print_write(f, curr_line)
            else:
                f.write('{}\n{}\n'.format(tokenizer.decode(sample),
                                          tokenizer.decode(captions[j].data)))


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    corr_list = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        corr_list.append(correct_k.item())
        #res.append(correct_k.mul_(100.0 / batch_size))
    return corr_list


def update_lr(optimizer):
    # For updating learning rate by 1/3 of current value
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 3


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_summary_stats(args, logger, f, top_acc, best_epoch_acc, lowest_loss,
                      best_epoch_loss, max_memory, time_all, top1_accuracies,
                      top5_accuracies, train_loss_avg, val_loss_avg):

    curr_line = '''\n{}
    Above configuration finished training successfully. 
    Best val accuracy: {}, at epoch no: {}/{}.
    Best val loss: {}, at epoch no: {}/{}.
    Highest reserved memory: {} (GB).
    Total time (loading, training and evaluation): {} seconds. Average: {} seconds.
    Time to reach top accuracy: {} seconds.\n'''.format(
        str(args), top_acc, best_epoch_acc, args.no_epochs, lowest_loss,
        best_epoch_loss, args.no_epochs, max_memory,
        time_all, time_all / args.no_epochs,
        best_epoch_acc * (time_all / args.no_epochs))
    print_write(f, curr_line)
    logger.info(curr_line)

    # contains the top1/5 accuracies for the validation after each epoch, and the last one for the test
    df_accuracies = pd.DataFrame(list(zip(top1_accuracies, top5_accuracies)))
    # contains the training and validation loss averages for each epoch
    df_losses = pd.DataFrame(list(zip(train_loss_avg, val_loss_avg)))

    df_metrics = pd.concat([df_accuracies, df_losses], axis=1)
    df_metrics.columns = [
        'top1_acc', 'top5_acc', 'train_loss_avg', 'val_loss_avg'
    ]
    df_metrics.to_csv(os.path.join(args.results_dir,
                                   '{}_metrics.csv'.format(args.run_name)),
                      sep=',',
                      header=True,
                      index=False)

    wandb.run.summary['Best top-1 accuracy'] = top_acc
    wandb.run.summary['Best epoch (acc)'] = best_epoch_acc
    wandb.run.summary['Best validation loss'] = lowest_loss
    wandb.run.summary['Best epoch (loss)'] = best_epoch_loss
    wandb.run.summary['Time total (s)'] = time_all
    wandb.run.summary['Average time per epoch (s)'] = time_all / args.no_epochs
    wandb.run.summary['Time to reach top accuracy'] = best_epoch_acc * (
        time_all / args.no_epochs)
    wandb.run.summary['Peak memory consumption (GB)'] = max_memory

    f.close()
    wandb.finish()


def save_checkpoints(args, model, epoch, curr_acc, top_acc, best_epoch_acc,
                     curr_val_loss, lowest_loss, best_epoch_loss):

    # Save the model checkpoint if the top1-acc is higher than current highest
    if curr_acc is not None and curr_acc > top_acc:
        torch.save(
            model.state_dict(),
            os.path.join(args.results_dir,
                         '{}_bestAccEpoch.ckpt'.format(args.run_name)))
        top_acc = curr_acc
        best_epoch_acc = epoch + 1

    # save if val loss is lower than current lowest and has reached cooldown period
    no_epochs_cd = args.mask_cd_percent * args.no_epochs
    if (curr_val_loss is not None and (curr_val_loss < lowest_loss)
            and (epoch > (args.no_epochs - no_epochs_cd))):
        torch.save(
            model.state_dict(),
            os.path.join(args.results_dir,
                         '{}_bestLossEpoch.ckpt'.format(args.run_name)))
        lowest_loss = curr_val_loss
        best_epoch_loss = epoch + 1

    # save each args.save_checkpoint_freq epochs
    # if (epoch + 1) % args.save_checkpoint_freq == 0:
    #     torch.save(
    #         model.state_dict(),
    #         os.path.join(args.results_dir,
    #                      '{}_epoch{}.ckpt'.format(args.run_name, epoch + 1)))

    # Saves model for last epoch regardless
    # (necessary for mlm versions since accuracy is not good metric for those)
    torch.save(
        model.state_dict(),
        os.path.join(args.results_dir,
                     '{}_lastEpoch.ckpt'.format(args.run_name)))

    return top_acc, best_epoch_acc, lowest_loss, best_epoch_loss


def vis_attention(args, image, outputs, att_mat, file_name_no_ext):

    #outputs = outputs.squeeze(0)
    #print(outputs.shape)
    #print(len(att_mat))
    #print(att_mat[0].shape)
    #print(outputs, att_mat)
    #print('logits_size and att_mat sizes: ', outputs.shape, att_mat.shape)

    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.shape)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    #print(att_mat.shape)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print('residual_att and aug_att_mat sizes: ', residual_att.shape, aug_att_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n],
                                           joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]  # last layer output attention map
    #print('joint_attentions and last layer (v) sizes: ', joint_attentions.shape, v.shape)
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    #print(mask.shape)
    mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
    #print(mask.shape)
    result = (mask * image).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(image)
    _ = ax2.imshow(result)

    print('-----')
    if not os.path.exists(os.path.join(args.results_dir, 'attention')):
        os.mkdir(os.path.join(args.results_dir, 'attention'))

    for idx in torch.topk(outputs, k=3).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        #print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))

    i = 0
    v = 0
    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
        result = (mask * image).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        title = 'AttentionMap_Layer{}'.format(i + 1)
        ax2.set_title(title)
        _ = ax1.imshow(image)
        _ = ax2.imshow(result)
        out_name = '{}_{}.jpg'.format(file_name_no_ext, title)
        plt.savefig(os.path.join(args.results_dir, 'attention', out_name))
        plt.close()
    i = 0
    v = 0
