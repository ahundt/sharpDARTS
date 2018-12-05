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
from model_search import Network
from architect import Architect
from PIL import Image
import random
from tqdm import tqdm
import dataset

parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset:\
                    cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers_of_cells', type=int, default=8, help='total number of cells in the whole network, default is 8 cells')
parser.add_argument('--layers_in_cells', type=int, default=4, help='total number of layers in each cell, aka number of steps')
parser.add_argument('--reduce_spacing', type=int, default=None, help='Number of layers between each reduction cell. 1 will mean all reduction cells.')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save, args.dataset)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_file_path = os.path.join(args.save, 'log.txt')
logger = utils.logging_setup(log_file_path)

def main():
  if not torch.cuda.is_available():
    logger.info('no gpu device available, exiting')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logger.info('gpu device = %d' % args.gpu)
  logger.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  number_of_classes = dataset.class_dict[args.dataset]
  in_channels = dataset.inp_channel_dict[args.dataset]
  model = Network(args.init_channels, number_of_classes, layers=args.layers_of_cells, criterion=criterion,
                  in_channels=in_channels, steps=args.layers_in_cells)
  model = model.cuda()
  logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # Get preprocessing functions (i.e. transforms) to apply on data
  train_transform, valid_transform = utils.get_data_transforms(args)

  # Get the training queue, select training and validation from training set
  train_queue, valid_queue = dataset.get_training_queues(args.dataset, train_transform, args.data, args.batch_size, args.train_portion)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args.momentum, args.weight_decay, args.arch_learning_rate, arch_weight_decay=args.arch_weight_decay)

  perfor = utils.Performance(os.path.join(args.save, 'architecture_performance_history.npy'))

  for epoch in tqdm(range(args.epochs)):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logger.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logger.info('genotype = %s', genotype)

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perfor)
    logger.info('train_acc %f', train_acc)

    perfor.save()

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logger.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perfor):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input_batch, target) in enumerate(tqdm(train_queue)):
    model.train()
    n = input_batch.size(0)

    input_batch = Variable(input_batch, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    # define validation loss for analyzing the importance of hyperparameters
    val_loss = architect.step(input_batch, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    # add current performance config into performance array
    perfor.update(model.alphas_normal, model.alphas_reduce, val_loss)

    optimizer.zero_grad()
    logits = model(input_batch)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

