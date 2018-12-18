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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from tqdm import tqdm
from Padam import Padam
import darts.cnn.model as model
import dataset
from learning_rate_schedulers import CosineWithRestarts


parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset:\
                    cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--partial', default=1/8, type=float, help='partially adaptive parameter p in Padam')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=700, help='num of training epochs')
parser.add_argument('--warm_restarts', type=int, default=10, help='warm restarts of cosine annealing')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='CHOKE_FLOOD', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  genotype = eval("genotypes.%s" % args.arch)
  number_of_classes = dataset.class_dict[args.dataset]
  in_channels = dataset.inp_channel_dict[args.dataset]

  cnn_model = model.NetworkCIFAR(
    args.init_channels, number_of_classes, args.layers,
    args.auxiliary, genotype, in_channels=in_channels)
  cnn_model = cnn_model.cuda()

  logger.info("param size = %fMB", utils.count_parameters_in_MB(cnn_model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  optimizer = Padam(cnn_model.parameters(), args.learning_rate, partial=args.partial, weight_decay=args.weight_decay)
  # optimizer = torch.optim.SGD(
  #     cnn_model.parameters(),
  #     args.learning_rate,
  #     momentum=args.momentum,
  #     weight_decay=args.weight_decay
  #     )

  # Get preprocessing functions (i.e. transforms) to apply on data
  train_transform, valid_transform = utils.get_data_transforms(args)

  # Get the training queue, use full training and test set
  train_queue, valid_queue = dataset.get_training_queues(
    args.dataset, train_transform, args.data, args.batch_size, train_proportion=1.0, train=False)

  scheduler = CosineWithRestarts(optimizer, t_max=float(args.warm_restarts), eta_min=float(args.learning_rate_min), factor=2)

  for epoch in tqdm(range(args.epochs), dynamic_ncols=True):
    scheduler.step()
    cnn_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, cnn_model, criterion, optimizer)

    valid_acc, valid_obj = infer(valid_queue, cnn_model, criterion)
    logger.info('epoch, %d, lr, %e, train_acc, %f, valid_acc, %f, train_loss, %f, valid_loss, %f',  epoch, scheduler.get_lr()[0], train_acc, valid_acc, train_obj, valid_obj)

    utils.save(cnn_model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, cnn_model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  cnn_model.train()

  progbar = tqdm(train_queue, dynamic_ncols=True)
  for step, (input_batch, target) in enumerate(progbar):
    input_batch = Variable(input_batch).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = cnn_model(input_batch)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(cnn_model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input_batch.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    progbar.set_description('loss: {0:9.5f}, top 1: {1:5.2f}, top 5: {2:5.2f}'.format(objs.avg, top1.avg, top5.avg))

  return top1.avg, objs.avg


def infer(valid_queue, cnn_model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  cnn_model.eval()

  with torch.no_grad():
    for step, (input_batch, target) in enumerate(tqdm(valid_queue, dynamic_ncols=True)):
      input_batch = Variable(input_batch).cuda()
      target = Variable(target).cuda(async=True)

      logits, _ = cnn_model(input_batch)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input_batch.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

    logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

