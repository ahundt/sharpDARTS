import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import copy
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import networkx as nx

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
from cosine_power_annealing import cosine_power_annealing
import matplotlib.pyplot as plt
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("Needs PyGraphviz or PyDotPlus to generate graph visualization")


parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset: cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-4, help='min learning rate')
parser.add_argument('--lr_power_annealing_exponent_order', type=float, default=2,
                    help='Cosine Power Annealing Schedule Base, larger numbers make '
                         'the exponential more dominant, smaller make cosine more dominant, '
                         '1 returns to standard cosine annealing.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful for restarts)')
parser.add_argument('--warmup_epochs', type=int, default=5, help='num of warmup training epochs')
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
parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, metavar='PATH', default='',
                    help='evaluate model at specified path on training, test, and validation datasets')
parser.add_argument('--load', type=str, default='',  metavar='PATH', help='load weights at specified location')
parser.add_argument('--load_args', type=str, default='',  metavar='PATH',
                    help='load command line args from a json file, this will override '
                         'all currently set args except for --evaluate, and arguments '
                         'that did not exist when the json file was originally saved out.')
parser.add_argument('--weighting_algorithm', type=str, default='scalar',
                    help='which operations to use, options are '
                         '"max_w" (1. - max_w + w) * op, and scalar (w * op)')
# TODO(ahundt) remove final path and switch back to genotype
parser.add_argument('--final_path', type=str, default=None, help='path for final model')
parser.add_argument('--load_genotype', type=str, default=None, help='Name of genotype to be used')
args = parser.parse_args()

args.arch = args.primitives + '-' + args.ops
# TODO(ahundt) enable --dataset flag, merge code from mixed_aux branch
args = utils.initialize_files_and_args(args, run_type='search')

logger = utils.logging_setup(args.log_file_path)

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
    final_path = None
    if args.final_path is not None:
      final_path = np.load(args.final_path)

    genotype = None
    if args.load_genotype is not None:
      genotype = getattr(genotypes, args.load_genotype)
    cnn_model = model_search.MultiChannelNetwork(
      args.init_channels, CIFAR_CLASSES, layers=args.layers_of_cells, criterion=criterion, steps=args.layers_in_cells,
      weighting_algorithm=args.weighting_algorithm, genotype=genotype)
    save_graph(cnn_model.G, os.path.join(args.save, 'network_graph.pdf'))
    if args.load_genotype is not None:
      # TODO(ahundt) support other batch shapes
      data_shape = [1, 3, 32, 32]
      batch = torch.zeros(data_shape)
      cnn_model(batch)
      logger.info("loaded genotype_raw_weights = " + str(cnn_model.genotype('raw_weights')))
      logger.info("loaded genotype_longest_path = " + str(cnn_model.genotype('longest_path')))
      logger.info("loaded genotype greedy_path = " + str(gen_greedy_path(cnn_model.G, strategy="top_down")))
      logger.info("loaded genotype greedy_path_bottom_up = " + str(gen_greedy_path(cnn_model.G, strategy="bottom_up")))
      # TODO(ahundt) support other layouts
  else:
    cnn_model = model_search.Network(
      args.init_channels, CIFAR_CLASSES, layers=args.layers_of_cells, criterion=criterion, steps=args.layers_in_cells,
      primitives=primitives, op_dict=op_dict, weights_are_parameters=args.no_architect, C_mid=args.mid_channels,
      weighting_algorithm=args.weighting_algorithm)
  cnn_model = cnn_model.cuda()
  logger.info("param size = %fMB", utils.count_parameters_in_MB(cnn_model))

  if args.load:
    logger.info('loading weights from: ' + args.load)
    utils.load(cnn_model, args.load)

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

  lr_schedule = cosine_power_annealing(
    epochs=args.epochs, max_lr=args.learning_rate, min_lr=args.learning_rate_min,
    warmup_epochs=args.warmup_epochs, exponent_order=args.lr_power_annealing_exponent_order)
  epochs = np.arange(args.epochs) + args.start_epoch
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
  #       optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  if args.no_architect:
    architect = None
  else:
    architect = Architect(cnn_model, args)

  epoch_stats = []

  stats_csv = args.epoch_stats_file
  stats_csv = stats_csv.replace('.json', '.csv')
  with tqdm(epochs, dynamic_ncols=True) as prog_epoch:
    best_valid_acc = 0.0
    best_epoch = 0
    # state_dict = {}
    # og_state_keys = set()
    # updated_state_keys = set()

    #saving state_dict for debugging weights by comparison
    # for key in cnn_model.state_dict():
    #   state_dict[key] = cnn_model.state_dict()[key].clone()
    #   # logger.info('layer = {}'.format(key))
    # logger.info('Total keys in state_dict = {}'.format(len(cnn_model.state_dict().keys())))
    # og_state_keys.update(cnn_model.state_dict().keys())
    best_stats = {}
    weights_file = os.path.join(args.save, 'weights.pt')
    for epoch, learning_rate in zip(prog_epoch, lr_schedule):
      # scheduler.step()
      # lr = scheduler.get_lr()[0]
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
      genotype = None
      if args.final_path is None:
        genotype = cnn_model.genotype()
        logger.info('genotype = %s', genotype)

      if not args.multi_channel:
        # the genotype is the alphas in the multi-channel case
        # print the alphas in other cases
        logger.info('alphas_normal = %s', cnn_model.arch_weights(0))
        logger.info('alphas_reduce = %s', cnn_model.arch_weights(1))

      # training
      train_acc, train_obj = train(train_queue, valid_queue, cnn_model, architect, criterion, optimizer, learning_rate)

      if args.multi_channel and args.final_path is None:
        # TODO(ahundt) remove final path and switch back to genotype, and save out raw weights plus optimal path
        optimal_path = nx.algorithms.dag.dag_longest_path(cnn_model.G)
        optimal_path_filename = os.path.join(args.save, 'longest_path_layer_sequence.npy')
        logger.info('Saving model layer sequence object: ' + str(optimal_path_filename))
        np.save(optimal_path_filename, optimal_path)
        graph_filename = os.path.join(args.save, 'network_graph_' + str(epoch) + '.graph')
        logger.info('Saving updated weight graph: ' + str(graph_filename))
        nx.write_gpickle(cnn_model.G, graph_filename)
        logger.info('optimal_path  : %s', optimal_path)
        capacity = nx.get_edge_attributes(cnn_model.G, "capacity")
        logger.info('capacity :%s', capacity)

      # for key in cnn_model.state_dict():
      #  updated_state_dict[key] = cnn_model.state_dict()[key].clone()


      # logger.info("gradients computed")
      # for name, parameter in cnn_model.named_parameters():
      #   if parameter.requires_grad:
      #     logger.info("{}  gradient  {}".format(name, parameter.grad.data.sum()))

      # updated_state_keys = set()
      # for key in state_dict:
      #   if not (state_dict[key] == updated_state_dict[key]).all():
      #     # logger.info('Update in {}'.format(key))
      #     updated_state_keys.add(key)
      # logger.info('Total updates = {}'.format(len(updated_state_keys)))
      # logger.info('Parameters not updated {}'.format(og_state_keys - updated_state_keys))

      # validation
      valid_acc, valid_obj = infer(valid_queue, cnn_model, criterion)

      if valid_acc > best_valid_acc:
        # new best epoch, save weights
        capacity = nx.get_edge_attributes(cnn_model.G, "capacity")
        weight = nx.get_edge_attributes(cnn_model.G, "weight")
        for u, v, d in cnn_model.G.edges(data=True):
          if u is not "Source" or v is not "Linear":
            if "capacity" in d:
              d['capacity'] = int(d['capacity']*1e+5)
            if "weight" in d:
              d['weight'] = int(d['weight']*1e+7)
        utils.save(cnn_model, weights_file)
        mincostFlow = nx.max_flow_min_cost(cnn_model.G, "Source", "Linear", weight='capacity', capacity='weight')
        new_mincost_flow = {}
        for key in mincostFlow:
          dic = mincostFlow[key]
          temp = {k: v for k, v in dic.items() if v != 0}
          if len(temp):
            new_mincost_flow[key] = temp
        capacity = nx.get_edge_attributes(cnn_model.G, "capacity")
        capacity = nx.get_edge_attributes(cnn_model.G, "weight")
        logger.info('capacity :%s', capacity)
        logger.info('weight :%s', weight)
        logger.info('mincostFlow  : %s', new_mincost_flow)
        mincostFlow_path_filename = os.path.join(args.save, 'micostFlow_path_layer_sequence.npy')
        np.save(mincostFlow_path_filename, new_mincost_flow)
        graph_filename = os.path.join(args.save, 'network_graph_best_valid' + str(epoch) + '.graph')
        logger.info('Saving updated weight graph: ' + str(graph_filename))
        best_epoch = epoch
        best_valid_acc = valid_acc
        prog_epoch.set_description(
            'Overview ***** best_epoch: {0} best_valid_acc: {1:.2f} ***** Progress'
            .format(best_epoch, best_valid_acc))

      logger.info('epoch, %d, train_acc, %f, valid_acc, %f, train_loss, %f, valid_loss, %f, lr, %e, best_epoch, %d, best_valid_acc, %f',
                  epoch, train_acc, valid_acc, train_obj, valid_obj, learning_rate, best_epoch, best_valid_acc)
      stats = {
        'epoch': epoch,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'train_loss': train_obj,
        'valid_loss': valid_obj,
        'lr': learning_rate,
        'best_epoch': best_epoch,
        'best_valid_acc': best_valid_acc,
        'genotype': str(genotype),
        'arch_weights': str(cnn_model.arch_weights)}
      epoch_stats += [copy.deepcopy(stats)]
      with open(args.epoch_stats_file, 'w') as f:
        json.dump(epoch_stats, f, cls=utils.NumpyEncoder)
      utils.list_of_dicts_to_csv(stats_csv, epoch_stats)

  # print the final model
  if args.final_path is None:
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
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

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
      input_batch = Variable(input_batch).cuda(non_blocking=True)
      target = Variable(target).cuda(non_blocking=True)

      logits = model(input_batch)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input_batch.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)
      progbar.set_description('Search Validation step: {0}, loss: {1:9.5f}, top 1: {2:5.2f} top 5: {3:5.2f} progress'.format(step, objs.avg, top1.avg, top5.avg))

  return top1.avg, objs.avg

def save_graph(G, file_name):
  pos=graphviz_layout(G, prog='dot')
  plt.figure(figsize=(160, 180))
  nx.draw_networkx_nodes(G, pos, node_shape="s",nodelist=G.nodes(),node_size=1000, linewidths=0.1, vmin=0, vmax=1, alpha=1)
  nx.draw_networkx_edges(G, pos, edgelist=G.edges(),width=1, edge_color="black", alpha=0.8)
  nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
  # figure(num=1, figsize=(100, 80), dpi=1000, facecolor='w', edgecolor='k')
  # plt.figure(1,figsize=(1200,1200))
  plt.axis('off')
  plt.tight_layout()
  plt.savefig(file_name)

def gen_greedy_path(G, strategy="top_down"):
  if strategy == "top_down":
    start_ = "Source"
    current_node = "Source"
    end_node = "Linear"
    new_G = G
  elif strategy == "bottom_up":
    start_ = "Linear"
    current_node = "Linear"
    end_node = "Source"
    new_G = G.reverse(copy=True)
  wt = 0
  node_list = []
  while current_node != end_node:
      neighbors = [n for n in new_G.neighbors(start_)]
      for nodes in neighbors:
          weight_ = new_G.get_edge_data(start_, nodes, "weight")
          # print(weight_)
          if len(weight_):
              weight_ = weight_["weight"]
          else:
              weight_ = 0
  #         print(weight_)
          if weight_ > wt:
              wt = weight_
              current_node = nodes
      node_list.append(current_node)
      # print("start",start_)
      # print(node)
      start_ = current_node
      wt = -1
  # print(node_list)
  if strategy == "bottom_up":
    node_list = node_list[::-1]
    node_list.append("Linear")
  return node_list


if __name__ == '__main__':
  main()
