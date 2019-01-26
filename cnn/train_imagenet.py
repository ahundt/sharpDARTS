import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkImageNet as Network
import train
import autoaugment
import operations

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='imagenet', help='which dataset, only option is imagenet')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--mid_channels', type=int, default=96, help='C_mid channels in choke SharpSepConv')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SHARP_DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
args = parser.parse_args()

args.save = 'eval-{}-{}-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.save, args.dataset, args.arch)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_file_path = os.path.join(args.save, 'log.txt')
logger = utils.logging_setup(log_file_path)
params_path = os.path.join(args.save, 'commandline_args.json')
with open(params_path, 'w') as f:
    json.dump(vars(args), f)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


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

  genotype = eval("genotypes.%s" % args.arch)
  cnn_model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, op_dict=op_dict, C_mid=args.mid_channels)
  if args.parallel:
    cnn_model = nn.DataParallel(cnn_model).cuda()
  else:
    cnn_model = cnn_model.cuda()

  logger.info("param size = %fMB", utils.count_parameters_in_MB(cnn_model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    cnn_model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      autoaugment.ImageNetPolicy(),
      # transforms.ColorJitter(
      #   brightness=0.4,
      #   contrast=0.4,
      #   saturation=0.4,
      #   hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  prog_epoch = tqdm(range(args.epochs), dynamic_ncols=True)
  best_valid_acc = 0.0
  best_epoch = 0
  best_stats = {}
  best_acc_top1 = 0
  for epoch in prog_epoch:
    scheduler.step()
    cnn_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train.train(train_queue, cnn_model, criterion, optimizer)
    stats = train.infer(valid_queue, cnn_model, criterion)

    is_best = False
    if stats['valid_acc'] > best_valid_acc:
      # new best epoch, save weights
      utils.save(cnn_model, weights_file)
      best_epoch = epoch
      best_valid_acc = stats['valid_acc']

      best_stats = stats
      best_stats['lr'] = scheduler.get_lr()[0]
      best_stats['epoch'] = best_epoch
      best_train_loss = train_obj
      best_train_acc = train_acc
      is_best = True

    logger.info('epoch, %d, train_acc, %f, valid_acc, %f, train_loss, %f, valid_loss, %f, lr, %e, best_epoch, %d, best_valid_acc, %f, ' + utils.dict_to_log_string(stats),
                epoch, train_acc, stats['valid_acc'], train_obj, stats['valid_loss'], scheduler.get_lr()[0], best_epoch, best_valid_acc)
    checkpoint = {
          'epoch': epoch,
          'state_dict': cnn_model.state_dict(),
          'best_acc_top1': best_valid_acc,
          'optimizer' : optimizer.state_dict(),
    }
    checkpoint.update(stats)
    utils.save_checkpoint(stats, is_best, args.save)

  best_epoch_str = utils.dict_to_log_string(best_stats, key_prepend='best_')
  logger.info(best_epoch_str)
  logger.info('Training of Final Model Complete! Save dir: ' + str(args.save))

if __name__ == '__main__':
  main()
