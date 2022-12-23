import time
import os
import logging
from statistics import mean

import torch
from torchsummary import summary
import wandb
from transformers import BertTokenizer

import utilities as utilities

logger = logging.getLogger(__name__)


def environment_loader(args, init=True):

    # Init logger
    if init:
        wandb.init(config=args)
        wandb.run.name = '{}'.format(args.run_name)
        file_name = '{}_log.txt'.format(args.run_name)
        f = open(os.path.join(args.results_dir, '{}'.format(file_name)),
                 'w',
                 buffering=1)
        utilities.misc.print_write(f, str(args))

    # Set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utilities.misc.set_seed(args.seed)

    # dataloader and train/test datasets
    if args.use_test_set:
        train_set, train_loader = utilities.data_selection_customize.load_data(
            args, split='train', label_csv_name='train.csv')
        _, test_loader = utilities.data_selection_customize.load_data(
            args, split='test', label_csv_name='test.csv')
    else:
        train_set, train_loader = utilities.data_selection_customize.load_data(
            args, split='train', label_csv_name=args.label_csv_name)
        test_loader = None
    args.num_classes = train_set.num_classes
    classid_classname_dic = train_set.classes

    steps_per_epoch = len(train_loader)
    total_steps = args.no_epochs * steps_per_epoch

    # tokenizer and mask scheduler
    if args.multimodal:
        if args.tokenizer == 'wp':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            args.vocab_size = tokenizer.vocab_size
        elif args.tokenizer == 'tag':
            tokenizer = utilities.custom_tokenizer.CustomTokenizer(
                vocab_path=os.path.join(args.dataset_path, 'labels',
                                        'vocab.pkl'),
                max_text_seq_len=args.max_text_seq_len)
            args.vocab_size = tokenizer.vocab_size
        if args.mask_schedule:
            mask_wu_steps = int(total_steps * args.mask_wu_percent)
            mask_cd_steps = int(total_steps * args.mask_cd_percent)
            mask_scheduler = utilities.scheduler.MasksSchedule(
                device=device,
                mask_schedule=args.mask_schedule,
                masking_behavior=args.masking_behavior,
                tokenizer=args.tokenizer,
                vocab_size=args.vocab_size,
                batch_size=args.batch_size,
                max_text_seq_len=args.max_text_seq_len,
                warmup_steps=mask_wu_steps,
                cooldown_steps=mask_cd_steps,
                total_steps=total_steps,
                cycles=.5)
        else:
            mask_scheduler = None
    else:
        mask_scheduler = None
        tokenizer = None
        args.vocab_size = None

    # model
    model = utilities.model_selection.load_model(args, device)
    if (args.model_name not in [
            'shallow', 'efficientnetb0', 'resnet18', 'resnet50', 'resnet152'
    ]) and init:
        utilities.misc.print_write(f, str(model.configuration))
    if (not args.interm_features_fc) and (not args.multimodal) and (init):
        summary(model, input_size=iter(train_loader).next()[0].shape[1:])

    # loss and optimizer
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = torch.optim.SGD(params_to_update,
                                lr=args.learning_rate,
                                momentum=0.9)

    if args.lr_scheduler == 'warmupCosine':
        lr_scheduler = utilities.scheduler.WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)
    else:
        lr_scheduler = None

    if init:
        return [
            f, device, train_set, train_loader, test_loader,
            classid_classname_dic, model, optimizer, lr_scheduler,
            mask_scheduler, tokenizer
        ]
    else:
        return [
            device, train_set, train_loader, test_loader,
            classid_classname_dic, model, optimizer, lr_scheduler,
            mask_scheduler, tokenizer
        ]


def train_one_epoch(args, f, epoch, global_step, model, device, tokenizer,
                    optimizer, mask_scheduler, lr_scheduler, train_loader,
                    train_loss_avg):

    criterion = torch.nn.CrossEntropyLoss()
    if args.mask_schedule:
        criterion_mlm = torch.nn.CrossEntropyLoss(reduction='none')

    model.train()
    current_losses = []
    steps_per_epoch = len(train_loader)

    for i, batch in enumerate(train_loader):

        if args.multimodal:
            images, labels, captions = batch
            captions = captions.squeeze(dim=1).to(device)
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # return updated caption tokens (and token labels for cross entropy) according to schedule
        if mask_scheduler is not None:
            captions_updated, labels_text = mask_scheduler.ret_mask(
                global_step, captions)
        else:
            labels_text = None

        # Forward pass
        if args.multimodal and args.mask_schedule and args.exclusion_loss:
            outputs, outputs_text, exclusion_loss = model(
                images, text=captions_updated)
        elif args.multimodal and args.mask_schedule:
            outputs, outputs_text = model(images, text=captions_updated)
        elif args.multimodal and args.exclusion_loss:
            outputs, exclusion_loss = model(images, text=captions)
        elif args.multimodal:
            outputs = model(images, text=captions)
        elif args.exclusion_loss:
            outputs, exclusion_loss = model(images)
        else:
            outputs = model(images)

        loss = criterion(outputs, labels)
        if labels_text is not None:
            loss = loss + criterion(outputs_text.transpose(1, 2), labels_text)
        if args.exclusion_loss:
            loss = loss - (args.exclusion_weight *
                           (args.temperature**2) * exclusion_loss)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        optimizer.zero_grad()

        current_losses.append(loss.item())

        # update global and return new masks according to schedule
        global_step[0] += 1

        # prints current set of results after each args.log_freq iterations
        if (i % args.log_freq) == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            curr_line = "Epoch [{}/{}], Step [{}/{}] Loss: {:.8f}, LR: {:.8f}\n".format(
                epoch + 1, args.no_epochs, i + 1, steps_per_epoch, loss.item(),
                curr_lr)
            utilities.misc.print_write(f, curr_line)
            wandb.log({
                'Training loss (step)': loss.item(),
                'Learning rate (current)': curr_lr
            })

            if labels_text is not None:
                utilities.misc.decode_text(f, tokenizer, outputs_text,
                                           captions, captions_updated,
                                           labels_text)

        if args.debugging and ((i + 1) % (args.log_freq * 3) == 0):
            break

    # Decay learning rate
    if not lr_scheduler:
        if (epoch + 1) % args.epoch_decay == 0:
            utilities.misc.update_lr(optimizer)

    # calculates mean of losses for current epoch and appends to list of avgs
    train_loss_avg.append(mean(current_losses))
    wandb.log({'Training loss (epoch)': mean(current_losses)})


def validate(args,
             f,
             global_step,
             model,
             device,
             tokenizer,
             loader,
             mask_scheduler,
             top1_accuracies,
             top5_accuracies,
             val_loss_avg=[],
             save_all_captions=False):
    # Test the model (validation set)
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # dropout probability goes to 0
    if save_all_captions and mask_scheduler is not None:
        file_name = '{}_captions.txt'.format(args.run_name)
        save_all_captions_file = open(os.path.join(args.results_dir,
                                                   '{}'.format(file_name)),
                                      'w',
                                      buffering=1)

    criterion = torch.nn.CrossEntropyLoss()
    if args.mask_schedule:
        criterion_mlm = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()
    with torch.no_grad():
        correct_1 = 0
        correct_5 = 0
        total = 0
        current_losses = []
        steps_per_epoch = len(loader)

        for i, batch in enumerate(loader):
            if args.multimodal:
                images, labels, captions = batch
                captions = captions.squeeze(dim=1).to(device)
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # return new masks and updated caption tokens according to schedule
            if mask_scheduler is not None:
                captions_updated, labels_text = mask_scheduler.ret_mask(
                    global_step, captions)
            else:
                labels_text = None

            # Forward pass
            if args.multimodal and args.mask_schedule and args.exclusion_loss:
                outputs, outputs_text, exclusion_loss = model(
                    images, text=captions_updated)
            elif args.multimodal and args.mask_schedule:
                outputs, outputs_text = model(images, text=captions_updated)
            elif args.multimodal and args.exclusion_loss:
                outputs, exclusion_loss = model(images, text=captions)
            elif args.multimodal:
                outputs = model(images, text=captions)
            elif args.exclusion_loss:
                outputs, exclusion_loss = model(images)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            if labels_text is not None:
                loss = loss + criterion(outputs_text.transpose(1, 2),
                                        labels_text)
            if args.exclusion_loss:
                loss = loss - (args.exclusion_weight *
                               (args.temperature**2) * exclusion_loss)

            current_losses.append(loss.item())

            # calculate top-k (1 and 5) accuracy
            total += labels.size(0)
            curr_corr_list = utilities.misc.accuracy(outputs.data, labels, (
                1,
                5,
            ))
            correct_1 += curr_corr_list[0]
            correct_5 += curr_corr_list[1]

            if i % args.log_freq == 0:
                curr_line = "Validation/Test Step [{}/{}] Loss: {:.8f}\n".format(
                    i + 1, steps_per_epoch, loss.item())
                utilities.misc.print_write(f, curr_line)

                if labels_text is not None:
                    utilities.misc.decode_text(f, tokenizer, outputs_text,
                                               captions, captions_updated,
                                               labels_text)

            if save_all_captions and (mask_scheduler is not None):
                utilities.misc.decode_text(save_all_captions_file,
                                           tokenizer,
                                           outputs_text,
                                           captions,
                                           captions_updated,
                                           labels_text,
                                           num_print=outputs_text.shape[0],
                                           save_all_captions=True)

            if args.debugging and ((i + 1) % (args.log_freq * 3) == 0):
                break

        # append avg val loss
        curr_val_loss = mean(current_losses)
        val_loss_avg.append(curr_val_loss)

        # compute epoch accuracy in percentages
        curr_top1_acc = 100 * correct_1 / total
        top1_accuracies.append(curr_top1_acc)
        curr_line = 'Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %'.format(
            curr_top1_acc)
        utilities.misc.print_write(f, curr_line)

        curr_top5_acc = 100 * correct_5 / total
        top5_accuracies.append(curr_top5_acc)
        curr_line = 'Val/Test Top-5 Accuracy of the model on the test images: {:.4f} %'.format(
            curr_top5_acc)
        utilities.misc.print_write(f, curr_line)

        wandb.log({
            "Epoch": len(top1_accuracies),
            "Val Accuracy Top-1": curr_top1_acc,
            "Val Accuracy Top-5": curr_top5_acc,
            "Val Loss": mean(current_losses)
        })

        return curr_top1_acc, curr_val_loss


def train_main(logger, args):

    time_start = time.time()

    (f, device, train_set, train_loader, test_loader, classid_classname_dic,
     model, optimizer, lr_scheduler, mask_scheduler,
     tokenizer) = environment_loader(args)

    # Train the model
    train_loss_avg = []
    val_loss_avg = []
    top1_accuracies = []
    top5_accuracies = []

    best_epoch_acc = 0
    best_epoch_loss = 0
    top_acc = 0
    lowest_loss = 1e6

    max_memory = 0
    global_step = [0]

    for epoch in range(args.no_epochs):

        train_one_epoch(args, f, epoch, global_step, model, device, tokenizer,
                        optimizer, mask_scheduler, lr_scheduler, train_loader,
                        train_loss_avg)

        curr_max_memory = torch.cuda.max_memory_reserved() / (1024**3)
        if max_memory < curr_max_memory:
            max_memory = curr_max_memory

        if ((epoch + 1) % args.save_checkpoint_freq == 0
                or epoch == args.no_epochs - 1):

            if args.use_test_set:
                curr_acc, curr_val_loss = validate(
                    args,
                    f,
                    global_step,
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    loader=test_loader,
                    mask_scheduler=mask_scheduler,
                    top1_accuracies=top1_accuracies,
                    top5_accuracies=top5_accuracies,
                    val_loss_avg=val_loss_avg)
            else:
                curr_acc, curr_val_loss = None, None

            # save checkpoints (last epoch, best accuracy, best val loss, each few epochs)
            # and update best metric
            top_acc, best_epoch_acc, lowest_loss, best_epoch_loss = \
                utilities.misc.save_checkpoints(
                    args, model, epoch, curr_acc, top_acc, best_epoch_acc,
                    curr_val_loss, lowest_loss, best_epoch_loss)

    if args.use_test_set:
        # validate on test set using last checkpoint and save captions
        # (assume last checkpoint gets best captions)
        curr_line = '\nUsing checkpoint from last epoch: {}.\n'.format(
            args.no_epochs)
        utilities.misc.print_write(f, curr_line)
        validate(args,
                 f,
                 global_step,
                 model=model,
                 device=device,
                 tokenizer=tokenizer,
                 loader=test_loader,
                 mask_scheduler=mask_scheduler,
                 top1_accuracies=top1_accuracies,
                 top5_accuracies=top5_accuracies,
                 save_all_captions=args.mask_schedule)

        time_end = time.time()
        time_all = time_end - time_start

        utilities.misc.log_summary_stats(args, logger, f, top_acc,
                                         best_epoch_acc, lowest_loss,
                                         best_epoch_loss, max_memory, time_all,
                                         top1_accuracies, top5_accuracies,
                                         train_loss_avg, val_loss_avg)


def main():

    logging.basicConfig(
        filename='logs.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
    )

    parser = utilities.misc.ret_args()
    parser.add_argument(
        '--label_csv_name',
        default='labels.csv',
        help='Name of the csv file that store label information')
    parser.add_argument('--use_test_set',
                        action='store_true',
                        help='Use test/validation set during training')
    args = parser.parse_args()

    if args.model_name == 'B_16' or args.model_name == 'L_16':
        args.patch_size = 16
    elif args.model_name == 'B_32' or args.model_name == 'L_32':
        args.patch_size = 32

    if not args.multimodal:
        args.max_text_seq_len = None
        args.mask_schedule = None
    elif (args.multimodal) and (not args.max_text_seq_len):
        args.max_text_seq_len = int(16)
    else:
        args.max_text_seq_len = int(args.max_text_seq_len)

    if args.exclusion_loss and not args.interm_features_fc:
        args.exclusion_loss = False

    args.run_name = (
        '{}_{}_image{}_batch{}_SGDlr{}_pt{}_seed{}_' +
        '{}_inter{}_mm{}_textLen{}_mask{}{}{}tokenizingshuf{}').format(
            args.dataset_name, args.model_name, args.image_size,
            args.batch_size, args.learning_rate, args.pretrained, args.seed,
            args.lr_scheduler, args.interm_features_fc, args.multimodal,
            args.max_text_seq_len, args.mask_schedule, args.masking_behavior,
            args.tokenizer, args.shuffle_tokens)

    logger.info(args)

    os.makedirs(args.results_dir, exist_ok=True)

    train_main(logger, args)


if __name__ == '__main__':
    main()
