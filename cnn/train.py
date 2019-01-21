import os
import sys
import time
import glob
import json
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from tqdm import tqdm

import genotypes
import operations
import cifar10_1

parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset:\
                    cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0000001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--partial', default=1/8, type=float, help='partially adaptive parameter p in Padam')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--warm_restarts', type=int, default=20, help='warm restarts of cosine annealing')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--mixed_auxiliary', action='store_true', default=False, help='Learn weights for auxiliary networks during training. Overrides auxiliary flag')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use cifar10 autoaugment https://arxiv.org/abs/1805.09501')
parser.add_argument('--random_eraser', action='store_true', default=False, help='use random eraser')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--ops', type=str, default='OPS', help='which operations to use, options are OPS and DARTS_OPS')
parser.add_argument('--primitives', type=str, default='PRIMITIVES',
                    help='which primitive layers to use inside a cell search space,'
                         ' options are PRIMITIVES and DARTS_PRIMITIVES')
parser.add_argument('--optimizer', type=str, default='sgd', help='which optimizer to use, options are padam and sgd')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save, args.dataset, args.arch)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_file_path = os.path.join(args.save, 'log.txt')
logger = utils.logging_setup(log_file_path)
params_path = os.path.join(args.save, 'commandline_args.json')
with open(params_path, 'w') as f:
    json.dump(vars(args), f)

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

  # load the correct ops dictionary
  op_dict_to_load = "operations.%s" % args.ops
  print('loading op dict: ' + str(op_dict_to_load))
  op_dict = eval(op_dict_to_load)
  operations.OPS = op_dict

  # load the correct primitives list
  primitives_to_load = "genotypes.%s" % args.primitives
  print('loading primitives: ' + str(primitives_to_load))
  primitives_list = eval(primitives_to_load)
  genotypes.PRIMITIVES = primitives_list

  CIFAR_CLASSES = 10

  genotype = eval("genotypes.%s" % args.arch)
  cnn_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  cnn_model = cnn_model.cuda()

  logger.info("param size = %fMB", utils.count_parameters_in_MB(cnn_model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      cnn_model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  prog_epoch = tqdm(range(args.epochs), dynamic_ncols=True)
  best_valid_acc = 0.0
  best_epoch = 0
  weights_file = os.path.join(args.save, 'weights.pt')
  for epoch in prog_epoch:
    scheduler.step()
    cnn_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, cnn_model, criterion, optimizer)

    valid_acc, valid_obj = infer(valid_queue, cnn_model, criterion)

    if valid_acc > best_valid_acc:
      # new best epoch, save weights
      utils.save(cnn_model, weights_file)
      best_epoch = epoch
      best_valid_acc = valid_acc
      best_valid_loss = valid_obj
      best_train_loss = train_obj
      best_train_acc = train_acc
      best_lr = scheduler.get_lr()[0]
    # else:
    #   # not best epoch, load best weights
    #   utils.load(cnn_model, weights_file)
    logger.info('epoch, %d, train_acc, %f, valid_acc, %f, train_loss, %f, valid_loss, %f, lr, %e, best_epoch, %d, best_valid_acc, %f',
                epoch, train_acc, valid_acc, train_obj, valid_obj, scheduler.get_lr()[0], best_epoch, best_valid_acc)

  if args.dataset == 'cifar10':
    # evaluate best model weights on cifar 10.1
    # https://github.com/modestyachts/CIFAR-10.1
    valid_data = cifar10_1.CIFAR10_1(root=args.data, train=False, download=True, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    # load the best model weights
    utils.load(cnn_model, weights_file)
    cifar10_1_valid_acc, cifar10_1_valid_loss = infer(valid_queue, cnn_model, criterion)
    # printout all stats from best epoch including cifar10.1
    logger.info('best_epoch, %d, best_train_acc, %f, best_valid_acc, %f, best_train_loss, %f, best_valid_loss, %f, lr, %e, best_epoch, %d, best_valid_acc, %f cifar10.1_valid_acc, %f, cifar10.1_valid_loss, %f',
                best_epoch, best_train_acc, best_valid_acc, train_obj, valid_obj, best_lr, best_epoch, best_valid_acc, cifar10_1_valid_acc, cifar10_1_valid_loss)
  logger.info('Training of Final Model Complete! Save dir: ' + str(args.save))


def train(train_queue, cnn_model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  cnn_model.train()

  progbar = tqdm(train_queue, dynamic_ncols=True)
  for step, (input_batch, target) in enumerate(progbar):
    input_batch = Variable(input_batch)
    target = Variable(target)
    if torch.cuda.is_available():
      input_batch = input_batch.cuda()
      target = target.cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = cnn_model(input_batch)
    loss = criterion(logits, target)
    if logits_aux is not None and args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight * loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(cnn_model.parameters(), args.grad_clip)
    # if cnn_model.auxs is not None:
    #   # clip the aux weights even more so they don't jump too quickly
    #   nn.utils.clip_grad_norm_(cnn_model.auxs.alphas, args.grad_clip/10)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input_batch.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    progbar.set_description('Training loss: {0:9.5f}, top 1: {1:5.2f}, top 5: {2:5.2f} progress'.format(objs.avg, top1.avg, top5.avg))

  return top1.avg, objs.avg


def infer(valid_queue, cnn_model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  cnn_model.eval()

  with torch.no_grad():
    progbar = tqdm(valid_queue, dynamic_ncols=True)
    for step, (input_batch, target) in enumerate(progbar):
      input_batch = Variable(input_batch).cuda()
      target = Variable(target).cuda(async=True)

      logits, _ = cnn_model(input_batch)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input_batch.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)
      progbar.set_description('Validation step: {0}, loss: {1:9.5f}, top 1: {2:5.2f} top 5: {3:5.2f} progress'.format(step, objs.avg, top1.avg, top5.avg))

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

