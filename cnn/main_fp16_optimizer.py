# source: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
# license: BSD 3-Clause
#
# to install apex:
# pip3 install --user --upgrade -e . --global-option="build_ext" --global-option="--cpp_ext" --global-option="--cuda_ext"
#
# ### Multi-process training with FP16_Optimizer, dynamic loss scaling
#     $ python3 -m torch.distributed.launch --nproc_per_node=2 main_fp16_optimizer.py --fp16 --b 256 --save `git rev-parse --short HEAD` --epochs 300 --dynamic-loss-scale --workers 14 --data /home/costar/datasets/imagenet/
#
# # note that --nproc_per_node is NUM_GPUS.
# # Can add --sync_bn to sync bachnorm values if batch size is "very small" but note this also reduces img/s by ~10%.

import argparse
import os
import shutil
import time
import glob
import json

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
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from model import NetworkImageNet as Network
from tqdm import tqdm
import dataset
import genotypes
import autoaugment
import operations
import utils
import warmup_scheduler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--data', type=str, default='../data', help='path to dataset', metavar='DIR')
parser.add_argument('--arch', '-a', metavar='ARCH', default='SHARP_DARTS',
                    # choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: SHARP_DARTS)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run (default: 300)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate based on autoaugment https://arxiv.org/pdf/1805.09501.pdf.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--learning_rate_min', type=float, default=0.00016, help='min learning rate')
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of epochs for warmup (default: 10)')
parser.add_argument('--warmup_lr_divisor', default=10, type=int, help='factor by which to reduce lr at warmup start (default: 10)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
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
parser.add_argument('--dataset', type=str, default='imagenet', help='which dataset, only option is imagenet')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use cifar10 autoaugment https://arxiv.org/abs/1805.09501')
parser.add_argument('--random_eraser', action='store_true', default=False, help='use random eraser')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

cudnn.benchmark = True

best_top1 = 0
args = parser.parse_args()
logger = None
DATASET_CHANNELS = dataset.inp_channel_dict[args.dataset]
# print('>>>>>>>DATASET_CHANNELS: ' + str(DATASET_CHANNELS))

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), DATASET_CHANNELS, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        # tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

# CLASSES = 1000

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.local_rank)

def main():
    global best_top1, args, logger

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()


    # workaround for directory creation and log files when run as multiple processes
    # args.save = 'eval-{}-{}-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save, args.dataset, args.arch)
    args.save = 'eval-{}-{}-{}-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save, 'imagenet', args.arch, args.gpu)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_file_path = os.path.join(args.save, 'log.txt')
    logger = utils.logging_setup(log_file_path)
    params_path = os.path.join(args.save, 'commandline_args.json')
    with open(params_path, 'w') as f:
        json.dump(vars(args), f)

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            logger.info("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # # load the correct ops dictionary
    op_dict_to_load = "operations.%s" % args.ops
    logger.info('loading op dict: ' + str(op_dict_to_load))
    op_dict = eval(op_dict_to_load)

    # load the correct primitives list
    primitives_to_load = "genotypes.%s" % args.primitives
    logger.info('loading primitives:' + primitives_to_load)
    primitives = eval(primitives_to_load)
    logger.info('primitives: ' + str(primitives))
    # create model
    genotype = eval("genotypes.%s" % args.arch)
    # get the number of output channels
    classes = dataset.class_dict[args.dataset]
    # create the neural network
    model = Network(args.init_channels, classes, args.layers, args.auxiliary, genotype, op_dict=op_dict, C_mid=args.mid_channels)
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
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    init_lr = args.lr / args.warmup_lr_divisor
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    epoch_count = args.epochs - args.start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epoch_count))
    scheduler = warmup_scheduler.GradualWarmupScheduler(
        optimizer, args.warmup_lr_divisor, args.warmup_epochs, scheduler)

    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_top1 = checkpoint['best_top1']
                model.load_state_dict(checkpoint['state_dict'])
                # An FP16_Optimizer instance's state dict internally stashes the master params.
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')

    # if(args.arch == "inception_v3"):
    #     crop_size = 299
    #     val_size = 320 # I chose this value arbitrarily, we can adjust.
    # else:
    #     crop_size = 224
    #     val_size = 256

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(crop_size),
    #         transforms.RandomHorizontalFlip(),
    #         autoaugment.ImageNetPolicy(),
    #         # transforms.ToTensor(),  # Too slow, moved to data_prefetcher()
    #         # normalize,
    #     ]))
    # val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(val_size),
    #         transforms.CenterCrop(crop_size)
    #     ]))

    # train_sampler = None
    # val_sampler = None
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True,
    #     sampler=val_sampler,
    #     collate_fn=fast_collate)

    if args.dataset == 'imagenet':
        collate_fn = fast_collate
        normalize_as_tensor = False
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
        normalize_as_tensor = True

    # Get preprocessing functions (i.e. transforms) to apply on data
    train_transform, valid_transform = utils.get_data_transforms(args, normalize_as_tensor=False)
    # Get the training queue, select training and validation from training set
    train_loader, val_loader = dataset.get_training_queues(
        args.dataset, train_transform, valid_transform, args.data,
        args.batch_size, train_proportion=1.0,
        collate_fn=fast_collate, distributed=args.distributed)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    prog_epoch = tqdm(range(args.start_epoch, args.epochs), dynamic_ncols=True)
    best_stats = {}
    stats = {}
    best_epoch = 0
    for epoch in prog_epoch:
        if args.distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        # if args.distributed:
            # train_sampler.set_epoch(epoch)

        scheduler.step()
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch, args)
        if args.prof:
            break
        # evaluate on validation set
        top1, val_stats = validate(val_loader, model, criterion, args)
        stats.update(train_stats)
        stats.update(val_stats)
        stats['lr'] = '{0:.5f}'.format(scheduler.get_lr()[0])

        # remember best top1 and save checkpoint
        if args.local_rank == 0:
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            stats['epoch'] = epoch + 1
            stats['best_top_1'] = '{0:.3f}'.format(best_top1)
            if is_best:
                best_epoch = epoch + 1
                best_stats = stats
            stats['best_epoch'] = best_epoch

            stats_str = utils.dict_to_log_string(stats)
            logger.info(stats_str)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_top_1': best_top1,
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict()
            }, is_best, path=args.save)
    stats_str = utils.dict_to_log_string(best_stats, key_prepend='best_')
    logger.info(stats_str)

class data_prefetcher():
    def __init__(self, loader, mean=None, std=None):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        mean = np.array(mean) * 255
        std = np.array(std) * 255
        self.mean = torch.tensor(mean).cuda().view(1,3,1,1)
        self.std = torch.tensor(std).cuda().view(1,3,1,1)
        if args.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            if args.fp16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, epoch, args):
    loader_len = len(train_loader)
    if loader_len < 2:
        raise ValueError('train_loader only supports 2 or more batches and loader_len: ' + str(loader_len))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    speed = AverageMeter()
    losses = AverageMeter()
    top1m = AverageMeter()
    top5m = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    prefetcher = data_prefetcher(train_loader, mean=args.mean, std=args.std)
    # if args.dataset == 'imagenet':
    #     # TODO(ahundt) debug why this special case is needed
    #     prefetcher = data_prefetcher(train_loader, mean=args.mean, std=args.std)
    # else:
    #     prefetcher = iter(train_loader)

    # logger.info('\n\n>>>>>>>>>>>>>>>>>>>prefetcher.len: ' + str(loader_len))
    input, target = prefetcher.next()
    # logger.info('\n\n>>>>>>>>>>>>>>>>>>>target.shape: ' + str(target.shape))
    i = -1
    if args.local_rank == 0:
        progbar = tqdm(total=len(train_loader))
    else:
        progbar = None
    while input is not None:
        i += 1
        # scheduler in main now adjusts the lr
        # adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        if args.prof:
            if i > 10:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # output = model(input)
        # loss = criterion(output, target)

        # note here the term output is equivalent to logits
        output, logits_aux = model(input)
        loss = criterion(output, target)
        if logits_aux is not None and args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        # measure accuracy and record loss
        top1f, top5f = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            top1f = reduce_tensor(top1f)
            top5f = reduce_tensor(top5f)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1m.update(to_python_float(top1f), input.size(0))
        top5m.update(to_python_float(top5f), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        input, target = prefetcher.next()

        if args.local_rank == 0:
            progbar.update()
        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            speed.update(args.world_size * args.batch_size / batch_time.val)
            progbar.set_description(
                #   'Epoch: [{0}][{1}/{2}]\t'
                  'Training (cur/avg)  '
                  'batch_t: {batch_time.val:.3f}/{batch_time.avg:.3f}, '
                  'img/s: {0:.3f}/{1:.3f}  '
                  'load_t: {data_time.val:.3f}/{data_time.avg:.3f}, '
                  'loss: {loss.val:.4f}/{loss.avg:.4f}, '
                  'top1: {top1.val:.3f}/{top1.avg:.3f}, '
                  'top5: {top5.val:.3f}/{top5.avg:.3f}, progress'.format(
                #    epoch, i, len(train_loader),
                   speed.val,
                   speed.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1m, top5=top5m))
    stats = {}
    prefix = 'train_'
    stats = get_stats(progbar, prefix, args, batch_time, data_time, top1m, top5m, losses, speed)
    return stats

def get_stats(progbar, prefix, args, batch_time, data_time, top1, top5, losses, speed):
    stats = {}
    if progbar is not None:
        stats = utils.tqdm_stats(progbar, prefix=prefix)
    stats.update({
        prefix + 'time_step_wall': '{0:.3f}'.format(args.world_size * args.batch_size / batch_time.avg),
        prefix + 'batch_time_one_gpu': '{0:.3f}'.format(batch_time.avg),
        prefix + 'data_time': '{0:.3f}'.format(data_time.avg),
        prefix + 'top1': '{0:.3f}'.format(top1.avg),
        prefix + 'top5': '{0:.3f}'.format(top5.avg),
        prefix + 'loss': '{0:.4f}'.format(losses.avg),
        prefix + 'images_per_second': '{0:.4f}'.format(speed.avg),
    })
    return stats


def validate(val_loader, model, criterion, args):
    loader_len = len(val_loader)
    if loader_len < 2:
        raise ValueError('val_loader only supports 2 or more batches and loader_len: ' + str(loader_len))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    speed = AverageMeter()
    losses = AverageMeter()
    top1m = AverageMeter()
    top5m = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader, mean=args.mean, std=args.std)
    input, target = prefetcher.next()
    i = -1
    if args.local_rank == 0:
        progbar = tqdm(total=loader_len)
    else:
        progbar = None
    while input is not None:
        i += 1
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.no_grad():
            # output = model(input)
            # loss = criterion(output, target)
            # note here the term output is equivalent to logits
            output, _ = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        top1f, top5f = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            top1f = reduce_tensor(top1f)
            top5f = reduce_tensor(top5f)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1m.update(to_python_float(top1f), input.size(0))
        top5m.update(to_python_float(top5f), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0:
            progbar.update()
        if args.local_rank == 0 and i % args.print_freq == 0:
            speed.update(args.world_size * args.batch_size / batch_time.val)
            progbar.set_description(
                # 'Test: [{0}/{1}]\t'
                  ' Validation (cur/avg)  '
                  'batch_t: {batch_time.val:.3f}/{batch_time.avg:.3f}, '
                  'img/s: {0:.3f}/{1:.3f}, '
                  'loss: {loss.val:.4f}/{loss.avg:.4f}, '
                  'top1: {top1.val:.3f}/{top1.avg:.3f}, '
                  'top5: {top5.val:.3f}/{top5.avg:.3f}, progress'.format(
                #    i, len(val_loader),
                   speed.val,
                   speed.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1m, top5=top5m))

        input, target = prefetcher.next()

    # logger.info(' * top1 {top1.avg:.3f} top5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))
    prefix = 'val_'
    stats = get_stats(progbar, prefix, args, batch_time, data_time, top1m, top5m, losses, speed)
    return top1m.avg, stats


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

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     logger.info("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
