import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import model_search
from architect import Architect
from PIL import Image
import random
from tqdm import tqdm
# import dataset
# from Padam import Padam
import json
# from learning_rate_schedulers import CosineWithRestarts
import operations
import genotypes
import dataset


parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset: cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--mid_channels', type=int, default=32, help='C_mid channels in choke SharpSepConv')
parser.add_argument('--layers_of_cells', type=int, default=8, help='total number of cells in the whole network, default is 8 cells')
parser.add_argument('--layers_in_cells', type=int, default=4,
                    help='Total number of nodes in each cell, aka number of steps,'
                         ' default is 4 nodes, which implies 8 ops')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use cifar10 autoaugment https://arxiv.org/abs/1805.09501')
parser.add_argument('--random_eraser', action='store_true', default=False, help='use random eraser')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--no_architect', action='store_true', default=False, help='directly train genotype parameters, disable architect.')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--multi_channel', action='store_true', default=False, help='perform multi channel search, a completely separate search space')
parser.add_argument('--ops', type=str, default='OPS', help='which operations to use, options are OPS and DARTS_OPS')
parser.add_argument('--primitives', type=str, default='PRIMITIVES',
                    help='which primitive layers to use inside a cell search space,'
                         ' options are PRIMITIVES and DARTS_PRIMITIVES')
args = parser.parse_args()

# TODO(ahundt) enable --dataset flag, merge code from mixed_aux branch
assert args.dataset == 'cifar10'
args.save = 'search-{}-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save, args.dataset)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_file_path = os.path.join(args.save, 'log.txt')
logger = utils.logging_setup(log_file_path)
params_path = os.path.join(args.save, 'commandline_args.json')
with open(params_path, 'w') as f:
    json.dump(vars(args), f)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logger.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logger.info('gpu device = %d' % args.gpu)
  logger.info("args = %s", args)

  # # load the correct ops dictionary
  op_dict_to_load = "operations.%s" % args.ops
  logger.info('loading op dict: ' + str(op_dict_to_load))
  op_dict = eval(op_dict_to_load)

  # load the correct primitives list
  primitives_to_load = "genotypes.%s" % args.primitives
  logger.info('loading primitives:' + primitives_to_load)
  primitives = eval(primitives_to_load)
  logger.info('primitives: ' + str(primitives))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  if args.multi_channel:
    cnn_model = model_search.MultiChannelNetwork(
      args.init_channels, CIFAR_CLASSES, layers=args.layers_of_cells, criterion=criterion, steps=args.layers_in_cells)
  else:
    cnn_model = model_search.Network(
      args.init_channels, CIFAR_CLASSES, layers=args.layers_of_cells, criterion=criterion, steps=args.layers_in_cells,
      primitives=primitives, op_dict=op_dict, weights_are_parameters=args.no_architect, C_mid=args.mid_channels)
  cnn_model = cnn_model.cuda()
  logger.info("param size = %fMB", utils.count_parameters_in_MB(cnn_model))

  optimizer = torch.optim.SGD(
      cnn_model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # Get preprocessing functions (i.e. transforms) to apply on data
  train_transform, valid_transform = utils.get_data_transforms(args)

  # Get the training queue, select training and validation from training set
  train_queue, valid_queue = dataset.get_training_queues(
    args.dataset, train_transform, valid_transform, args.data, args.batch_size, args.train_portion,
    search_architecture=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  if args.no_architect:
    architect = None
  else:
    architect = Architect(cnn_model, args)

  prog_epoch = tqdm(range(args.epochs), dynamic_ncols=True)
  best_valid_acc = 0.0
  best_epoch = 0
  for epoch in prog_epoch:
    scheduler.step()
    lr = scheduler.get_lr()[0]

    genotype = cnn_model.genotype()
    logger.info('genotype = %s', genotype)

    if not args.multi_channel:
      # the genotype is the alphas in the multi-channel case
      # print the alphas in other cases
      logger.info('alphas_normal = %s', cnn_model.arch_weights(0))
      logger.info('alphas_reduce = %s', cnn_model.arch_weights(1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, cnn_model, architect, criterion, optimizer, lr)

    # validation
    valid_acc, valid_obj = infer(valid_queue, cnn_model, criterion)

    if valid_acc > best_valid_acc:
      # new best epoch, save weights
      utils.save(cnn_model, os.path.join(args.save, 'weights.pt'))
      best_epoch = epoch
      best_valid_acc = valid_acc

    logger.info('epoch, %d, train_acc, %f, valid_acc, %f, train_loss, %f, valid_loss, %f, lr, %e, best_epoch, %d, best_valid_acc, %f',
                epoch, train_acc, valid_acc, train_obj, valid_obj, scheduler.get_lr()[0], best_epoch, best_valid_acc)

  # print the final model
  genotype = cnn_model.genotype()
  logger.info('genotype = %s', genotype)
  logger.info('Search for Model Complete! Save dir: ' + str(args.save))


def train(train_queue, valid_queue, cnn_model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  progbar = tqdm(train_queue, dynamic_ncols=True)
  for step, (input_batch, target) in enumerate(progbar):
    cnn_model.train()
    n = input_batch.size(0)

    input_batch = Variable(input_batch, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda(async=True)
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    # define validation loss for analyzing the importance of hyperparameters
    if architect is not None:
      val_loss = architect.step(input_batch, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = cnn_model(input_batch)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(cnn_model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    progbar.set_description('Training loss: {0:9.5f}, top 1: {1:5.2f}, top 5: {2:5.2f} progress'.format(objs.avg, top1.avg, top5.avg))

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    progbar = tqdm(valid_queue, dynamic_ncols=True)
    for step, (input_batch, target) in enumerate(progbar):
      input_batch = Variable(input_batch).cuda(async=True)
      target = Variable(target).cuda(async=True)

      logits = model(input_batch)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input_batch.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)
      progbar.set_description('Search Validation step: {0}, loss: {1:9.5f}, top 1: {2:5.2f} top 5: {3:5.2f} progress'.format(step, objs.avg, top1.avg, top5.avg))

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

