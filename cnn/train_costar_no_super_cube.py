# source: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
# license: BSD 3-Clause
#
# to install apex:
# pip3 install --user --upgrade -e . --global-option="build_ext" --global-option="--cpp_ext" --global-option="--cuda_ext"
#
# ### Multi-process training with FP16_Optimizer, dynamic loss scaling
#     $ python3 -m torch.distributed.launch --nproc_per_node=2 train_costar.py --fp16 --b 256 --save `git rev-parse --short HEAD` --epochs 300 --dynamic-loss-scale --workers 14 --data ~/.keras/datasets/costar_block_stacking_dataset_v0.4
#
# # note that --nproc_per_node is NUM_GPUS.
# # Can add --sync_bn to sync bachnorm values if batch size is "very small" but note this also reduces img/s by ~10%.
#
# Example command:
#
#    export CUDA_VISIBLE_DEVICES="2" && python3 train_costar.py --auxiliary --cutout --batch_size 128 --epochs 200 --save `git rev-parse --short HEAD` --arch SHARP_DARTS --mid_channels 32 --init_channels 36 --wd 0.0003 --lr_power_annealing_exponent_order 2 --learning_rate_min 0.0005 --learning_rate 0.05
import argparse
import os
import shutil
import time
import glob
import json
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision

import numpy as np
import random

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

try:
    import costar_dataset
except ImportError:
    ImportError('The costar dataset is not available. '
                'See https://github.com/ahundt/costar_dataset for details')

from model import NetworkImageNet, NetworkCOSTAR
from tqdm import tqdm
import dataset
import genotypes
import autoaugment
import operations
import utils
import warmup_scheduler
from cosine_power_annealing import cosine_power_annealing
from costar_baseline_model import NetworkResNetCOSTAR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--data', type=str, default='~/.keras/datasets/costar_block_stacking_dataset_v0.4',
                    help='path to dataset', metavar='DIR')
parser.add_argument('--arch', '-a', metavar='ARCH', default='SHARP_DARTS',
                    # choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: SHARP_DARTS)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run (default: 300)')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful for restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning_rate', dest='learning_rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate based on autoaugment https://arxiv.org/pdf/1805.09501.pdf.  Will be scaled by <global batch size>/256: args.learning_rate = args.learning_rate*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--learning_rate_min', type=float, default=0.00016, help='min learning rate')
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of epochs for warmup (default: 10)')
parser.add_argument('--warmup_lr_divisor', default=10, type=int, help='factor by which to reduce lr at warmup start (default: 10)')
parser.add_argument('--lr_power_annealing_exponent_order', type=float, default=10,
                    help='Cosine Power Annealing Schedule Base, larger numbers make '
                         'the exponential more dominant, smaller make cosine more dominant.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', dest='weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--restart_lr', action='store_true',
                    help='Used in conjunction with --resume, '
                         'this will restart the lr curve as if it was epoch 1, '
                         'but otherwise retain your current epoch count.')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('--deterministic', action='store_true', default=False)

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')

parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--ops', type=str, default='OPS', help='which operations to use, options are OPS and DARTS_OPS')
parser.add_argument('--primitives', type=str, default='PRIMITIVES',
                    help='which primitive layers to use inside a cell search space,'
                         ' options are PRIMITIVES and DARTS_PRIMITIVES')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--mid_channels', type=int, default=96, help='C_mid channels in choke SharpSepConv')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use cifar10 autoaugment https://arxiv.org/abs/1805.09501')
parser.add_argument('--random_eraser', action='store_true', default=False, help='use random eraser')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=112, help='cutout length')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, metavar='PATH', default='',
                    help='evaluate model at specified path on training, test, and validation datasets')
parser.add_argument('--load', type=str, default='',  metavar='PATH', help='load weights at specified location')
parser.add_argument('--load_args', type=str, default='',  metavar='PATH',
                    help='load command line args from a json file, this will override '
                         'all currently set args except for --evaluate, and arguments '
                         'that did not exist when the json file was originally saved out.')
# CoSTAR BSD specific arguments
parser.add_argument('--dataset', type=str, default='stacking', help='which dataset, only option is stacking')
parser.add_argument('--version', type=str, default='v0.4', help='the CoSTAR BSD version to use. Defaults to "v0.4"')
parser.add_argument('--set_name', type=str, default='blocks_only',
                    help='which set to use in the CoSTAR BSD. Options are "blocks_only" or "blocks_with_plush_toy". '
                         'Defaults to "blocks_only"')
parser.add_argument('--subset_name', type=str, default='success_only',
                    help='which subset to use in the CoSTAR BSD. Options are "success_only", '
                         '"error_failure_only", "task_failure_only", or "task_and_error_failure". Defaults to "success_only"')
parser.add_argument('--feature_mode', type=str, default='all_features',
                    help='which feature mode to use. Options are "translation_only", "rotation_only", "stacking_reward", '
                         'or the default "all_features"')
parser.add_argument('--num_images_per_example', type=int, default=200,
                    help='Number of times an example is visited per epoch. Default value is 200. Since the image for each visit to an '
                         'example is randomly chosen, and since the number of images in an example is different, we simply visit each '
                         'example multiple times according to this number to ensure most images are visited.')
parser.add_argument('--cart_weight', type=float, default=0.7,
                    help='the weight for the cartesian error. In validation, the metric to determine whether a run is good is '
                         'comparing the weighted sum of cart_weight*cart_error+(1-cart_weight)*angle_error. Defaults to 0.7 '
                         'because translational error is more important than rotational error.')
parser.add_argument('--abs_cart_error_output_csv_name', type=str, default='abs_cart_error.csv',
                    help='the output csv file name for the absolute cartesian error of ALL samples in ALL epochs. '
                         'Actual output file will have train_/val_/test_ prefix')
parser.add_argument('--abs_angle_error_output_csv_name', type=str, default='abs_angle_error.csv',
                    help='the output csv file name for the absolute cartesian error of ALL samples in ALL epochs. '
                         'Actual output file will have train_/val_/test_ prefix')

cudnn.benchmark = True

best_combined_error = float('inf')
args = parser.parse_args()
logger = None

DATASET_CHANNELS = dataset.costar_inp_channel_dict[args.feature_mode]
VECTOR_SIZE = dataset.costar_vec_size_dict[args.feature_mode]


def fast_collate(batch):
    data, targets = zip(*batch)

    image_0 = torch.tensor([img[0] for img in data])
    image_n = torch.tensor([img[1] for img in data])
    vector = torch.tensor([img[2] for img in data])

    return torch.cat((image_0, image_n), dim=1), vector, torch.tensor(targets)


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.local_rank)


def main():
    global best_combined_error, args, logger

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # commented because it is now set as an argparse param.
    # args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    # note the gpu is used for directory creation and log files
    # which is needed when run as multiple processes
    args = utils.initialize_files_and_args(args)
    logger = utils.logging_setup(args.log_file_path)

    # # load the correct ops dictionary
    op_dict_to_load = "operations.%s" % args.ops
    logger.info('loading op dict: ' + str(op_dict_to_load))
    op_dict = eval(op_dict_to_load)

    # load the correct primitives list
    primitives_to_load = "genotypes.%s" % args.primitives
    logger.info('loading primitives:' + primitives_to_load)
    primitives = eval(primitives_to_load)
    logger.info('primitives: ' + str(primitives))
    # get the number of output channels
    classes = dataset.costar_class_dict[args.feature_mode]

    # create model
    genotype = eval("genotypes.%s" % args.arch)
    # create the neural network
    # model = NetworkImageNet(args.init_channels, classes, args.layers, args.auxiliary, genotype, in_channels=DATASET_CHANNELS, op_dict=op_dict, C_mid=args.mid_channels)
    model = NetworkCOSTAR(args.init_channels, classes, args.layers, args.auxiliary, genotype, vector_size=VECTOR_SIZE, op_dict=op_dict, C_mid=args.mid_channels)

    model.drop_path_prob = 0.0
    # if args.pretrained:
    #     logger.info("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     logger.info("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()

    if args.sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    # NOTE(rexxarchl): MSLE loss, indicated as better for rotation in costar_hyper/costar_block_stacking_train_regression.py
    #                  is not available in PyTorch by default

    # Scale learning rate based on global batch size
    args.learning_rate = args.learning_rate * float(args.batch_size * args.world_size)/256.
    init_lr = args.learning_rate / args.warmup_lr_divisor
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # epoch_count = args.epochs - args.start_epoch
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epoch_count))
    # scheduler = warmup_scheduler.GradualWarmupScheduler(
    #     optimizer, args.warmup_lr_divisor, args.warmup_epochs, scheduler)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        if args.evaluate:
            args.resume = args.evaluate
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                if 'best_combined_error' in checkpoint:
                    best_combined_error = checkpoint['best_combined_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                # TODO(ahundt) make sure scheduler loading isn't broken
                if 'lr_scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['lr_scheduler'])
                elif 'lr_schedule' in checkpoint:
                    lr_schedule = checkpoint['lr_schedule']
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Get preprocessing functions (i.e. transforms) to apply on data
    # normalize_as_tensor = False because we normalize and convert to a
    # tensor in our custom prefetching function, rather than as part of
    # the transform preprocessing list.
    # train_transform, valid_transform = utils.get_data_transforms(args, normalize_as_tensor=False)
    train_transform = valid_transform = None  # NOTE(rexxarchl): data transforms are not applicable for CoSTAR BSD at the moment
    # Get the training queue, select training and validation from training set
    train_loader, val_loader = dataset.get_training_queues(
        args.dataset, train_transform, valid_transform, args.data,
        args.batch_size, train_proportion=1.0,
        collate_fn=fast_collate,
        distributed=args.distributed,
        num_workers=args.workers,
        costar_set_name=args.set_name, costar_subset_name=args.subset_name,
        costar_feature_mode=args.feature_mode, costar_version=args.version, costar_num_images_per_example=args.num_images_per_example,
        costar_output_shape=(224, 224, 3), costar_random_augmentation=None, costar_one_hot_encoding=True)

    if args.evaluate:
        test_loader = dataset.get_costar_test_queue(
                args.data, costar_set_name=args.set_name, costar_subset_name=args.subset_name,
                costar_feature_mode=args.feature_mode, costar_version=args.version, costar_num_images_per_example=args.num_images_per_example,
                costar_output_shape=(224, 224, 3), costar_random_augmentation=None, costar_one_hot_encoding=True)
        validate(test_loader, model, criterion, args, prefix='test_')
        return

    lr_schedule = cosine_power_annealing(
        epochs=args.epochs, max_lr=args.learning_rate, min_lr=args.learning_rate_min,
        warmup_epochs=args.warmup_epochs, exponent_order=args.lr_power_annealing_exponent_order,
        restart_lr=args.restart_lr)
    epochs = np.arange(args.epochs) + args.start_epoch

    stats_csv = args.epoch_stats_file
    stats_csv = stats_csv.replace('.json', '.csv')
    with tqdm(epochs, dynamic_ncols=True, disable=args.local_rank != 0, leave=False) as prog_epoch:
        best_stats = {}
        stats = {}
        epoch_stats = []
        best_epoch = 0
        for epoch, learning_rate in zip(prog_epoch, lr_schedule):
            if args.distributed and train_loader.sampler is not None:
                train_loader.sampler.set_epoch(int(epoch))
            # if args.distributed:
                # train_sampler.set_epoch(epoch)
            # update the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            # scheduler.step()
            model.drop_path_prob = args.drop_path_prob * float(epoch) / float(args.epochs)
            # train for one epoch
            train_stats = train(train_loader, model, criterion, optimizer, int(epoch), args)
            if args.prof:
                break
            # evaluate on validation set
            combined_error, val_stats = validate(val_loader, model, criterion, args)
            stats.update(train_stats)
            stats.update(val_stats)
            # stats['lr'] = '{0:.5f}'.format(scheduler.get_lr()[0])
            stats['lr'] = '{0:.5f}'.format(learning_rate)
            stats['epoch'] = epoch

            # remember best combined_error and save checkpoint
            if args.local_rank == 0:
                is_best = combined_error < best_combined_error
                best_combined_error = min(combined_error, best_combined_error)
                stats['best_combined_error'] = '{0:.3f}'.format(best_combined_error)
                if is_best:
                    best_epoch = epoch
                    best_stats = copy.deepcopy(stats)
                stats['best_epoch'] = best_epoch

                stats_str = utils.dict_to_log_string(stats)
                logger.info(stats_str)
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_combined_error': best_combined_error,
                    'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler.state_dict()
                    'lr_schedule': lr_schedule,
                    'stats': best_stats
                }, is_best, path=args.save)
                prog_epoch.set_description(
                    'Overview ***** best_epoch: {0} best_valid_combined_error: {1:.2f} ***** Progress'
                    .format(best_epoch, best_combined_error))
            epoch_stats += [copy.deepcopy(stats)]
            with open(args.epoch_stats_file, 'w') as f:
                json.dump(epoch_stats, f, cls=utils.NumpyEncoder)
            utils.list_of_dicts_to_csv(stats_csv, epoch_stats)
        stats_str = utils.dict_to_log_string(best_stats, key_prepend='best_')
        logger.info(stats_str)
        with open(args.stats_file, 'w') as f:
            arg_dict = vars(args)
            arg_dict.update(best_stats)
            json.dump(arg_dict, f, cls=utils.NumpyEncoder)
        with open(args.epoch_stats_file, 'w') as f:
            json.dump(epoch_stats, f, cls=utils.NumpyEncoder)
        utils.list_of_dicts_to_csv(stats_csv, epoch_stats)
        logger.info('Training of Final Model Complete! Save dir: ' + str(args.save))


class data_prefetcher():
    def __init__(self, loader, in_channels, cutout=False, cutout_length=112, cutout_cuts=2):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        cutout_dtype = np.float32

        self.cutout = None
        if cutout:
            self.cutout = utils.BatchCutout(cutout_length, cutout_cuts, dtype=cutout_dtype)
        self.preload()

    def preload(self):
        try:
            self.next_input_img, self.next_input_vec, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input_img = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input_img = self.next_input_img.cuda(non_blocking=True).float()
            self.next_input_vec = self.next_input_vec.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).float()
            if self.cutout is not None:
                # TODO(ahundt) Fix performance of this cutout call, it makes batch loading time go from 0.001 seconds to 0.05 seconds.
                self.next_input_img = self.cutout(self.next_input_img)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_img = self.next_input_img
        input_vec = self.next_input_vec
        target = self.next_target
        self.preload()
        return input_img, input_vec, target


def train(train_loader, model, criterion, optimizer, epoch, args):
    loader_len = len(train_loader)
    if loader_len < 2:
        raise ValueError('train_loader only supports 2 or more batches and loader_len: ' + str(loader_len))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    speed = AverageMeter()
    losses = AverageMeter()
    abs_cart_m = AverageMeter()
    abs_angle_m = AverageMeter()
    sigmoid = torch.nn.Sigmoid()

    # switch to train mode
    model.train()
    end = time.time()
    prefetcher = data_prefetcher(train_loader, in_channels=DATASET_CHANNELS, cutout=args.cutout, cutout_length=args.cutout_length)

    cart_error, angle_error = [], []
    input_img, input_vec, target = prefetcher.next()
    batch_size = input_img.size(0)
    i = -1
    if args.local_rank == 0:
        progbar = tqdm(total=len(train_loader), leave=False, dynamic_ncols=True)
    else:
        progbar = None
    while input_img is not None:
        i += 1
        # scheduler in main now adjusts the lr
        # adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        if args.prof:
            if i > 10:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # note here the term output is equivalent to logits
        output, logits_aux = model(input_img, input_vec)
        output = sigmoid(output)
        loss = criterion(output, target)
        if logits_aux is not None and args.auxiliary:
            logits_aux = sigmoid(logits_aux)
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        # measure accuracy and record loss
        with torch.no_grad():
            output_np = output.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            batch_abs_cart_distance, batch_abs_angle_distance = accuracy(output_np, target_np)
            abs_cart_f, abs_angle_f = np.mean(batch_abs_cart_distance), np.mean(batch_abs_angle_distance)
            cart_error.extend(batch_abs_cart_distance)
            angle_error.extend(batch_abs_angle_distance)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            abs_cart_f = reduce_tensor(abs_cart_f)
            abs_angle_f = reduce_tensor(abs_angle_f)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss, batch_size)
        abs_cart_m.update(abs_cart_f, batch_size)
        abs_angle_m.update(abs_angle_f, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        input_img, input_vec, target = prefetcher.next()

        if args.local_rank == 0:
            progbar.update()
        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            speed.update(args.world_size * args.batch_size / batch_time.val)
            progbar.set_description(
                #   'Epoch: [{0}][{1}/{2}]\t'
                  'Train (cur/avg)  '
                  'batch_t: {batch_time.val:.3f}/{batch_time.avg:.3f}, '
                  'img/s: {0:.1f}/{1:.1f}  '
                  'load_t: {data_time.val:.3f}/{data_time.avg:.3f}, '
                  'loss: {loss.val:.4f}/{loss.avg:.4f}, '
                  'cart: {abs_cart.val:.2f}/{abs_cart.avg:.2f}, '
                  'angle: {abs_angle.val:.2f}/{abs_angle.avg:.2f}, prog'.format(
                #    epoch, i, len(train_loader),
                   speed.val,
                   speed.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, abs_cart=abs_cart_m, abs_angle=abs_angle_m))
    stats = {}
    prefix = 'train_'
    if args.feature_mode != 'rotation_only':  # translation_only or all_features: save cartesian csv
        utils.list_to_csv(os.path.join(args.save, prefix + args.abs_cart_error_output_csv_name),
                          cart_error)
    if args.feature_mode != 'translation_only':  # rotation_only or all_features: save angle csv
        utils.list_to_csv(os.path.join(args.save, prefix + args.abs_angle_error_output_csv_name),
                          angle_error)
    stats = get_stats(progbar, prefix, args, batch_time, data_time, abs_cart_m, abs_angle_m, losses, speed)
    if progbar is not None:
        progbar.close()
        del progbar
    return stats


def get_stats(progbar, prefix, args, batch_time, data_time, abs_cart, abs_angle, losses, speed):
    stats = {}
    if progbar is not None:
        stats = utils.tqdm_stats(progbar, prefix=prefix)
    stats.update({
        prefix + 'time_step_wall': '{0:.3f}'.format(args.world_size * args.batch_size / batch_time.avg),
        prefix + 'batch_time_one_gpu': '{0:.3f}'.format(batch_time.avg),
        prefix + 'data_time': '{0:.3f}'.format(data_time.avg),
        prefix + 'abs_cart': '{0:.3f}'.format(abs_cart.avg),
        prefix + 'abs_angle': '{0:.3f}'.format(abs_angle.avg),
        prefix + 'loss': '{0:.4f}'.format(losses.avg),
        prefix + 'images_per_second': '{0:.4f}'.format(speed.avg),
    })
    return stats


def validate(val_loader, model, criterion, args, prefix='val_'):
    loader_len = len(val_loader)
    if loader_len < 2:
        raise ValueError('val_loader only supports 2 or more batches and loader_len: ' + str(loader_len))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    speed = AverageMeter()
    losses = AverageMeter()
    abs_cart_m = AverageMeter()
    abs_angle_m = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    cart_error, angle_error = [], []
    prefetcher = data_prefetcher(val_loader, in_channels=DATASET_CHANNELS)
    input_img, input_vec, target = prefetcher.next()
    batch_size = input_img.size(0)
    i = -1
    if args.local_rank == 0:
        progbar = tqdm(total=loader_len)
    else:
        progbar = None
    while input_img is not None:
        i += 1
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.no_grad():
            # output = model(input)
            # loss = criterion(output, target)
            # note here the term output is equivalent to logits
            output, _ = model(input_img, input_vec)
            loss = criterion(output, target)

        # measure accuracy and record loss
        batch_abs_cart_distance, batch_abs_angle_distance = accuracy(output.data.cpu().numpy(), target.data.cpu().numpy())
        abs_cart_f, abs_angle_f = np.mean(batch_abs_cart_distance), np.mean(batch_abs_angle_distance)
        cart_error.extend(batch_abs_cart_distance)
        angle_error.extend(batch_abs_angle_distance)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            abs_cart_f = reduce_tensor(abs_cart_f)
            abs_angle_f = reduce_tensor(abs_angle_f)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss, batch_size)
        abs_cart_m.update(abs_cart_f, batch_size)
        abs_angle_m.update(abs_angle_f, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0:
            progbar.update()
        if args.local_rank == 0 and i % args.print_freq == 0:
            speed.update(args.world_size * args.batch_size / batch_time.val)
            progbar.set_description(
                # 'Test: [{0}/{1}]\t'
                  'Valid (cur/avg)  '
                  'batch_t: {batch_time.val:.3f}/{batch_time.avg:.3f}, '
                  'img/s: {0:.1f}/{1:.1f}, '
                  'loss: {loss.val:.4f}/{loss.avg:.4f}, '
                  'abs_cart: {abs_cart.val:.2f}/{abs_cart.avg:.2f}, '
                  'abs_angle: {abs_angle.val:.2f}/{abs_angle.avg:.2f}, prog'.format(
                #    i, len(val_loader),
                   speed.val,
                   speed.avg,
                   batch_time=batch_time, loss=losses,
                   abs_cart=abs_cart_m, abs_angle=abs_angle_m))

        input_img, input_vec, target = prefetcher.next()

    # logger.info(' * combined_error {combined_error.avg:.3f} top5 {top5.avg:.3f}'
    #       .format(combined_error=combined_error, top5=top5))
    if args.feature_mode != 'rotation_only':  # translation_only or all_features: save cartesian csv
        utils.list_to_csv(os.path.join(args.save, prefix + args.abs_cart_error_output_csv_name),
                          cart_error)
    if args.feature_mode != 'translation_only':  # rotation_only or all_features: save angle csv
        utils.list_to_csv(os.path.join(args.save, prefix + args.abs_angle_error_output_csv_name),
                          angle_error)
    stats = get_stats(progbar, prefix, args, batch_time, data_time, abs_cart_m, abs_angle_m, losses, speed)
    if progbar is not None:
        progbar.close()
        del progbar

    # Return the weighted sum of absolute cartesian and angle errors as the metric
    return (args.cart_weight * abs_cart_m.avg + (1-args.cart_weight) * abs_angle_m.avg), stats


def save_checkpoint(state, is_best, path='', filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    new_filename = os.path.join(path, filename)
    torch.save(state, new_filename)
    if is_best:
        shutil.copyfile(new_filename, os.path.join(path, best_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.learning_rate*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     logger.info("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the absolute cartesian and angle distance between output and target"""
    batch_size, out_channels = target.shape

    if out_channels == 3:  # xyz
        # Format into [batch, 8] by adding fake rotations
        fake_rotation = np.zeros([batch_size, 5], dtype=np.float32)
        target = np.concatenate((target, fake_rotation), 1)
        output = np.concatenate((output, fake_rotation), 1)
    elif out_channels == 5:  # aaxyz_nsc
        # Format into [batch, 8] by adding fake translations
        fake_translation = torch.zeros([batch_size, 3], dtype=np.float32)
        target = np.concatenate((fake_translation, target), 1)
        output = np.concatenate((fake_translation, output), 1)
    elif out_channels == 8:  # xyz + aaxyz_nsc
        pass  # Do nothing
    else:
        raise ValueError("accuracy: unknown number of output channels: {}".format(out_channels))

    abs_cart_distance = costar_dataset.cart_error(target, output)
    abs_angle_distance = costar_dataset.angle_error(target, output)

    return abs_cart_distance, abs_angle_distance


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
