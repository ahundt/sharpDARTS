# Example commands to run:
#
#    python3 visualize_whole_network.py --multi_channel
#
#    python3 visualize_whole_network.py --dataset imagenet --arch SHARP_DARTS --auxiliary
#
# Set matplotlib backend to Agg
# *MUST* be done BEFORE importing hiddenlayer or libs that import matplotlib
import matplotlib
matplotlib.use("Agg")

import os
import torch
# import networkx
import model_search
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

from model import NetworkImageNet as NetworkImageNet
from model import NetworkCIFAR as NetworkCIFAR
from tqdm import tqdm
import dataset
import genotypes
import autoaugment
import operations
import utils
import warmup_scheduler
from cosine_power_annealing import cosine_power_annealing
# requires https://github.com/waleedka/hiddenlayer
import hiddenlayer as hl


parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--arch', '-a', metavar='ARCH', default='multi_channel',
                    # choices=model_names,
                    help='model architecture: (default: multi_channel, other options are SHARP_DARTS and  DARTS). '
                         'multi_channel is for multi channel search, a completely separate search space.')
parser.add_argument('--ops', type=str, default='OPS', help='which operations to use, options are OPS and DARTS_OPS')
parser.add_argument('--primitives', type=str, default='PRIMITIVES',
                    help='which primitive layers to use inside a cell search space,'
                         ' options are PRIMITIVES and DARTS_PRIMITIVES')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--mid_channels', type=int, default=96, help='C_mid channels in choke SharpSepConv')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset, imagenet or cifar10')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
args = parser.parse_args()
if args.arch is not 'multi_channel':
  # # load the correct ops dictionary
  op_dict_to_load = "operations.%s" % args.ops
  print('loading op dict: ' + str(op_dict_to_load))
  op_dict = eval(op_dict_to_load)
  # load the correct primitives list
  primitives_to_load = "genotypes.%s" % args.primitives
  print('loading primitives:' + primitives_to_load)
  primitives = eval(primitives_to_load)
  print('primitives: ' + str(primitives))
  # create model
  genotype = eval("genotypes.%s" % args.arch)
  # get the number of output channels
  classes = dataset.class_dict[args.dataset]
  # create the neural network
  print('initializing module')
if args.arch == 'multi_channel':
    cnn_model = model_search.MultiChannelNetwork(always_apply_ops=True, layers=6, steps=3, visualization=True)
elif args.dataset == 'imagenet':
    cnn_model = NetworkImageNet(args.init_channels, classes, args.layers, args.auxiliary, genotype, op_dict=op_dict, C_mid=args.mid_channels)
    # workaround for graph generation limitations
    cnn_model.drop_path_prob = torch.zeros(1)
else:
    cnn_model = NetworkCIFAR(args.init_channels, classes, args.layers, args.auxiliary, genotype, op_dict=op_dict, C_mid=args.mid_channels)
    # workaround for graph generation limitations
    cnn_model.drop_path_prob = torch.zeros(1)

transforms = [
  hl.transforms.Fold('MaxPool3x3 > Conv1x1 > BatchNorm', 'ResizableMaxPool', 'ResizableMaxPool'),
  hl.transforms.Fold('MaxPool > Conv > BatchNorm', 'ResizableMaxPool', 'ResizableMaxPool'),
  hl.transforms.Fold('Relu > Conv > Conv > BatchNorm', 'ReluSepConvBn'),
  hl.transforms.Fold('ReluSepConvBn > ReluSepConvBn', 'SharpSepConv', 'SharpSepConv'),
  hl.transforms.Fold('Relu > Conv > BatchNorm', 'ReLUConvBN'),
  hl.transforms.Fold('Relu > Conv1x1 > BatchNorm', 'ReLUConv1x1BN'),
  hl.transforms.Prune('Constant'),
  hl.transforms.Prune('Gather'),
  hl.transforms.Prune('Unsqueeze'),
  hl.transforms.Prune('Concat'),
  hl.transforms.Prune('Shape'),
# Fold repeated blocks
  hl.transforms.FoldDuplicates(),
  hl.transforms.Fold('Relu > Conv > BatchNorm', 'ReLUConvBN'),
  hl.transforms.Fold('Relu > Conv1x1 > BatchNorm', 'ReLUConv1x1BN'),
]
print('building graph')
# WARNING: the code may hang here. These are instructions for a workaround:
# First install hiddenlayer from source:
#
#     cd ~/src
#     git clone https://github.com/waleedka/hiddenlayer.git
#     cd hiddenlayer
#     pip3 install --user --upgrade -e .
#
# Next open the file /hiddenlayer/hiddenlayer/pytorch_builder.py
#
# change:
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
# to
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.RAW)
#
# The graph is very large so building the graph will take a long time.
# Note that at the time of writing the graph algorithms can't handle multiplying by a constant.
# Instead, I added if statements that skip the weight component if it is in visualization mode.
#
# For progress bars go back to /hiddenlayer/hiddenlayer/pytorch_builder.py:
# at the top add:
#     import tqdm as tqdm
#
# Then in:
#
#     def import_graph()
#
# find all instances of:
#
#    torch_graph.nodes()
#
# and replace with:
#
#    tqdm(torch_graph.nodes())
#
if args.dataset == 'imagenet':
  input_batch = torch.zeros([2, 3, 224, 224])
elif args.dataset == 'cifar10':
  input_batch = torch.zeros([2, 3, 32, 32])
# print(input_batch)
cnn_graph = hl.build_graph(cnn_model, input_batch, transforms=transforms)
output_file = os.path.expanduser('~/src/darts/cnn/' + args.arch + '_network.pdf')
print('build complete, saving: ' + output_file)
cnn_graph.save(output_file)
print('save complete')