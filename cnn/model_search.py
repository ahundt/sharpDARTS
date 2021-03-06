import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES, MULTICHANNELNET_PRIMITIVES
from genotypes import Genotype
import networkx as nx
from networkx.readwrite import json_graph
import json
import time
import genotype_extractor


class MixedOp(nn.Module):

  def __init__(self, C, stride, primitives=None, op_dict=None, weighting_algorithm=None):
    """ Perform a mixed forward pass incorporating multiple primitive operations like conv, max pool, etc.

    # Arguments

      primitives: the list of strings defining the operations to choose from.
      op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict.
    """
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self._stride = stride
    if primitives is None:
          primitives = PRIMITIVES
    self._primitives = primitives
    if op_dict is None:
          op_dict = operations.OPS
    for primitive in primitives:
      op = op_dict[primitive](C, C, stride, False)
      # op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
      self._weighting_algorithm = weighting_algorithm

  def forward(self, x, weights):
    # result = 0
    # print('-------------------- forward')
    # print('weights shape: ' + str(len(weights)) + ' ops shape: ' + str(len(self._ops)))
    # for i, (w, op) in enumerate(zip(weights, self._ops)):
    #   print('w shape: ' + str(w.shape) + ' op type: ' + str(type(op)) + ' i: ' + str(i) + ' self._primitives[i]: ' + str(self._primitives[i]) + 'x size: ' + str(x.size()) + ' stride: ' + str(self._stride))
    #   op_out = op(x)
    #   print('op_out size: ' + str(op_out.size()))
    #   result += w * op_out
    # return result
    # apply all ops with intensity corresponding to their weight
    if self._weighting_algorithm is None or self._weighting_algorithm == 'scalar':
      return sum(w * op(x) for w, op in zip(weights, self._ops))
    elif self._weighting_algorithm == 'max_w':
      max_w = torch.max(weights)
      return sum((1. - max_w + w) * op(x) for w, op in zip(weights, self._ops))
    else:
      raise ValueError('MixedOP(): Unsupported weighting algorithm: ' + str(self._weighting_algorithm) +
                       ' try "scalar" or "max_w"')


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives=None, op_dict=None, weighting_algorithm=None):
    """Create a searchable cell representing multiple architectures.

    The Cell class in model.py is the equivalent for a single architecture.

    # Arguments
      steps: The number of primitive operations in the cell,
        essentially the number of low level layers.
      multiplier: The rate at which the number of channels increases.
      op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict.
    """
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev is None:
      self.preprocess0 = operations.Identity()
    elif reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, stride=2, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, primitives, op_dict, weighting_algorithm=weighting_algorithm)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C=16, num_classes=10, layers=8, criterion=None, steps=4, multiplier=4, stem_multiplier=3,
               in_channels=3, primitives=None, op_dict=None, C_mid=None, weights_are_parameters=False,
               weighting_algorithm=None):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    if criterion is None:
      self._criterion = nn.CrossEntropyLoss()
    else:
      self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._weights_are_parameters = weights_are_parameters
    if primitives is None:
      primitives = PRIMITIVES
    self.primitives = primitives

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                  reduction, reduction_prev, primitives, op_dict,
                  weighting_algorithm=weighting_algorithm)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(self.primitives)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    if self._weights_are_parameters:
      # in simpler training modes the weights are just regular parameters
      self.alphas_normal = torch.nn.Parameter(self.alphas_normal)
      self.alphas_reduce = torch.nn.Parameter(self.alphas_reduce)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def arch_weights(self, stride_idx):
    weights_softmax_view = self._arch_parameters[stride_idx]
    # apply softmax and convert to an indexable view
    weights = F.softmax(weights_softmax_view, dim=-1)
    return weights

  def genotype(self, skip_primitive='none'):
    '''
    Extract the genotype, or specific connections within a cell, as encoded by the weights.
    # Arguments
        skip_primitives: hack was added by DARTS to temporarily workaround the
            'strong gradient' problem identified in the sharpDARTS paper https://arxiv.org/abs/1903.09900,
            set skip_primitive=None to not skip any primitives.
    '''
    gene_normal = genotype_extractor.parse_cell(
      F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
      primitives=self.primitives, steps=self._steps, skip_primitive=skip_primitive)
    gene_reduce = genotype_extractor.parse_cell(
      F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
      primitives=self.primitives, steps=self._steps, skip_primitive=skip_primitive)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat,
      layout='cell',
    )
    return genotype

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

class MultiChannelNetwork(nn.Module):
  """
    This class is used to perform Differentiable Grid Search using a set of primitives of your choice (refer to genotypes.py and operations.py for 
    more information on the different types of primitives available and their definition).
    For more information about the search please refer to section 4 of the paper https://arxiv.org/abs/1903.09900
  """

  def __init__(self, C=32, num_classes=10, layers=6, criterion=None, steps=5, multiplier=4, stem_multiplier=3,
               in_channels=3, final_linear_filters=768, always_apply_ops=False, visualization=False, primitives=None, op_dict=None,
               weighting_algorithm=None, genotype=None):
    """ C is the mimimum number of channels. Layers is how many output scaling factors and layers should be in the network.
        op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict.

    """
    super(MultiChannelNetwork, self).__init__()
    self._C = C
    if genotype is not None:
      # TODO(ahundt) We shouldn't be using arrays here, we should be using actual genotype objects.
      self._genotype = np.array(genotype)
    else:
      self._genotype = genotype
    self._num_classes = num_classes
    if layers % 2 == 1:
      raise ValueError('MultiChannelNetwork layers option must be even, got ' + str(layers))
    self._layers = layers // 2
    if criterion is None:
      self._criterion = nn.CrossEntropyLoss()
    else:
      self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._always_apply_ops = always_apply_ops
    self._visualization = visualization
    self._weighting_algorithm = weighting_algorithm

    if primitives is None:
        primitives = MULTICHANNELNET_PRIMITIVES
    if op_dict is None:
        op_dict = operations.MULTICHANNELNET_OPS

    self.primitives = primitives
    self.op_dict = op_dict

    self.normal_index = 0
    self.reduce_index = 1
    self.layer_types = 2
    self.strides = np.array([self.normal_index, self.reduce_index])
    # 5 is a reasonable number
    self.C_start = int(np.log2(C))
    self.C_end = self.C_start + steps
    print('c_start: ' + str(self.C_start) + ' c_end: ' + str(self.C_end))
    self.Cs = np.array(np.exp2(np.arange(self.C_start, self.C_end)), dtype='int')
    # $ print(Cs)
    # [ 32.  64. 128. 256. 512.]
    self.C_size = len(self.Cs)
    C_in, C_out = np.array(np.meshgrid(self.Cs, self.Cs, indexing='ij'), dtype='int')
    # $ print(C_in)
    # [[ 32.  32.  32.  32.  32.]
    #  [ 64.  64.  64.  64.  64.]
    #  [128. 128. 128. 128. 128.]
    #  [256. 256. 256. 256. 256.]
    #  [512. 512. 512. 512. 512.]]
    # $ print(C_out)
    # [[ 32.  64. 128. 256. 512.]
    #  [ 32.  64. 128. 256. 512.]
    #  [ 32.  64. 128. 256. 512.]
    #  [ 32.  64. 128. 256. 512.]
    #  [ 32.  64. 128. 256. 512.]]
    
    # Switching to primitives.
    # self.op_types = [operations.SharpSepConv, operations.ResizablePool]

    self.stem = nn.ModuleList()
    self.G = nx.DiGraph()
    self.G.add_node("Source")
    self.G.nodes["Source"]['demand'] = -1
    for i, c in enumerate(self.Cs):
      s = nn.Sequential(
        nn.Conv2d(int(in_channels), int(c), 3, padding=1, bias=False),
        nn.BatchNorm2d(c)
      )
      self.G.add_edge("Source", "Conv3x3_"+str(i))
      self.G["Source"]["Conv3x3_"+str(i)]["weight"] = 600
      self.G.add_node("Conv3x3_"+str(i))
      self.G.add_node("BatchNorm_"+str(i))
      self.G.add_edge("Conv3x3_"+str(i), "BatchNorm_"+str(i))
      self.stem.append(s)
    for layer_idx in range(self._layers):
        for stride_idx in self.strides:
            for C_out_idx in range(self.C_size):
                out_node = 'layer_'+str(layer_idx)+'_add_'+'c_out_'+str(self.Cs[C_out_idx])+'_stride_' + str(stride_idx+1)
                self.G.add_node(out_node)
    self.op_grid = nn.ModuleList()
    for layer_idx in range(self._layers):
      stride_modules = nn.ModuleList()
      for stride_idx in self.strides:
        in_modules = nn.ModuleList()
        for C_in_idx in range(self.C_size):
          out_modules = nn.ModuleList()
          # print('init layer: ' + str(layer_idx) + ' stride: ' + str(stride_idx+1) + ' c_in: ' + str(self.Cs[C_in_idx]))
          for C_out_idx in range(self.C_size):
            out_node = 'layer_'+str(layer_idx)+'_add_'+'c_out_'+str(self.Cs[C_out_idx])+'_stride_' + str(stride_idx+1)
            type_modules = nn.ModuleList()

            # switching to primitives
            # for OpType in self.op_types:
            for primitive_idx, primitive in enumerate(self.primitives):
                cin = C_in[C_in_idx][C_out_idx]
                cout = C_out[C_in_idx][C_out_idx]
                # print('cin: ' + str(cin) + ' cout: ' + str(cout))
                name = 'layer_' + str(layer_idx) + '_stride_' + str(stride_idx+1) + '_c_in_' + str(self.Cs[C_in_idx]) + '_c_out_' + str(self.Cs[C_out_idx]) + '_op_type_' + str(primitive) + '_opid_' + str(primitive_idx)
                self.G.add_node(name)
                if layer_idx == 0 and stride_idx == 0:
                    self.G.add_edge("BatchNorm_"+str(C_in_idx), name)
                elif stride_idx > 0 or layer_idx == 0:
                    self.G.add_edge('layer_' + str(layer_idx)+'_add_' + 'c_out_'+str(self.Cs[C_in_idx])+'_stride_' + str(stride_idx), name)
                else:
                    self.G.add_edge('layer_' + str(layer_idx-1)+'_add_' + 'c_out_'+str(self.Cs[C_in_idx])+'_stride_' + str(self.strides[-1] + 1), name)
                self.G.add_edge(name, out_node)
                # op = OpType(int(cin), int(cout), kernel_size=3, stride=int(stride_idx + 1))
                op = self.op_dict[primitive](int(cin), int(cout), int(stride_idx + 1), False)
                # Consistent with MixedOp
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(int(cout), affine=False))
                # Decreasing feature maps so that output is as expected.
                if 'none' in primitive or ('skip_connect' in primitive and stride_idx == 0):
                    op = nn.Sequential(op, nn.Conv2d(int(cin), int(cout), 1))
                type_modules.append(op)
            out_modules.append(type_modules)
          in_modules.append(out_modules)
        # op grid is stride_modules
        stride_modules.append(in_modules)
      self.op_grid.append(stride_modules)

    self.base = nn.ModuleList()
    self.G.add_node("add-SharpSep")
    self.time_between_layers = AverageMeter()

    for c in self.Cs:
      self.G.add_node("SharpSepConv" + str(c))
      out_node = 'layer_'+str(self._layers-1)+'_add_'+'c_out_'+str(c)+'_stride_' + str(self.strides[-1] + 1)
      self.G.add_edge("SharpSepConv" + str(c), "add-SharpSep")
      self.G.add_edge(out_node, "SharpSepConv" + str(c))
      self.base.append(operations.SharpSepConv(int(c), int(final_linear_filters), 3))
    # TODO(ahundt) there should be one more layer of normal convolutions to set the final linear layer size
    # C_in will be defined by the previous layer's c_out
    self.arch_weights_shape = [len(self.strides), self._layers, self.C_size, self.C_size, len(self.primitives)]
    # number of weights total
    self.weight_count = np.prod(self.arch_weights_shape)
    # number of weights in a softmax call
    self.softmax_weight_count = np.prod(self.arch_weights_shape[2:])
    # minimum score for a layer to continue being trained
    self.min_score = float(1 / (self.softmax_weight_count * self.softmax_weight_count))

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(final_linear_filters, num_classes)
    self.G.add_node("global_pooling")
    self.G.add_edge("add-SharpSep", "global_pooling")
    self.G.add_node("Linear")
    self.G.nodes["Linear"]['demand'] = 1
    self.G.add_edge("global_pooling", "Linear")
    self.G["global_pooling"]["Linear"]["weight"] = 800

    print("Saving graph...")
    nx.write_gpickle(self.G, "network_test.graph")

    if not self._visualization:
      self._initialize_alphas(genotype)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input_batch):

    self.C_size = len(self.Cs)
    s0s = [[], [None] * self.C_size, [None] * self.C_size]
    for i, C_in in enumerate(self.Cs):
      # Make the set of features with different numbers of channels.
      s0s[0] += [self.stem[i](input_batch)]

    # calculate weights, there are two weight views according to stride
    weight_views = []
    if not self._visualization:
      for stride_idx in self.strides:
        # ops are stored as layer, stride, cin, cout, num_layer_types
        # while weights are ordered stride_index, layer, cout, num_layer_types
        # first exclude the stride_idx because we already know that
        weight_views += [self.arch_weights(stride_idx)]
    # Duplicate s0s to account for 2 different strides
    # s0s += [[]]
    # s1s = [None] * layers + 1

    # computing capacity during model eval
    if self.training is False:
        # time_between_layers = AverageMeter()
        end_time = time.time()
    for layer in range(self._layers):
      # layer is how many times we've called everything, i.e. the number of "layers"
      # this is different from the number of layer types which is len([SharpSepConv, ResizablePool]) == 2
      layer_st_time = time.time()
      for stride_idx in self.strides:
        stride = 1 + stride_idx
        # we don't pass the gradient along max_w because it is the weight for a different operation.
        # TODO(ahundt) is there a better way to create this variable without gradients & reallocating repeatedly?
        # max_w = torch.Variable(torch.max(weight_views[stride_idx][layer, :, :, :]), requires_grad=False).cuda()
        # find the maximum comparable weight, copy it and make sure we don't pass gradients along that path
        if not self._visualization and self._weighting_algorithm is not None and self._weighting_algorithm == 'max_w':
          max_w = torch.max(weight_views[stride_idx][layer, :, :, :])
        for C_out_idx, C_out in enumerate(self.Cs):
          # take all the layers with the same output so we can sum them
          # print('forward layer: ' + str(layer) + ' stride: ' + str(stride) + ' c_out: ' + str(self.Cs[C_out_idx]))
          out_node = 'layer_'+str(layer)+'_add_'+'c_out_'+str(C_out)+'_stride_' + str(stride_idx+1)
          c_outs = []
          # compute average time when validating model.
          if self.training is False:
            self.time_between_layers.update(time.time() - end_time)
            time_in_layers = AverageMeter()
          for C_in_idx, C_in in enumerate(self.Cs):
            for primitive_idx, primitive in enumerate(self.primitives):

              if self.training is False:
                op_st_time = time.time()
              # get the specific weight for this op
              name = 'layer_' + str(layer) + '_stride_' + str(stride_idx+1) + '_c_in_' + str(C_in) + '_c_out_' + str(C_out) + '_op_type_' + str(primitive) + '_opid_' + str(primitive_idx)
              if not self._visualization:
                w = weight_views[stride_idx][layer, C_in_idx, C_out_idx, primitive_idx]
                # self.G.add_edge(name, out_node, {weight: w})
                self.G[name][out_node]["weight"] = float(w.clone().cpu().detach().numpy())
                self.G[name][out_node]["weight_int"] = int(float(w.clone().cpu().detach().numpy()) * 1e+5)
              # print('w weight_views[stride_idx][layer, C_in_idx, C_out_idx, op_type_idx]: ' + str(w))
              # apply the operation then weight, equivalent to
              # w * op(input_feature_map)
              # TODO(ahundt) fix conditionally evaluating calls with high ratings, there is currently a bug
              if self._always_apply_ops or w > self.min_score:
                # only apply an op if weight score isn't too low: w > 1/(N*N)
                # x = 1 - max_w + w so that max_w gets a score of 1 and everything else gets a lower score accordingly.
                s = s0s[stride_idx][C_in_idx]
                if s is not None:
                  if not self._visualization:
                    if self._weighting_algorithm is None or self._weighting_algorithm == 'scalar':
                      x = w * self.op_grid[layer][stride_idx][C_in_idx][C_out_idx][primitive_idx](s)
                    elif self._weighting_algorithm == 'max_w':
                      # print(name)
                      # print(s.size())
                      x = (1. - max_w + w) * self.op_grid[layer][stride_idx][C_in_idx][C_out_idx][primitive_idx](s)
                      # self.G[name][out_node]["weight"] = (1. - max_w + w)
                    else:
                      raise ValueError(
                        'MultiChannelNetwork.forward(): Unsupported weighting algorithm: ' +
                        str(self._weighting_algorithm) + ' try "scalar" or "max_w"')
                  else:
                    # doing visualization, skip the weights
                    x = self.op_grid[layer][stride_idx][C_in_idx][C_out_idx][primitive_idx](s)
                  c_outs += [x]
                # compute average time when validating model.
                if self.training is False:
                    time_in_layers.update(time.time() - op_st_time)
                    self.G[name][out_node]["capacity"] = time_in_layers.avg + self.time_between_layers.avg
                    end_time = time.time()

          # only apply updates to layers of sufficient quality
          if c_outs:
            # print('combining c_outs forward layer: ' + str(layer) + ' stride: ' + str(stride) + ' c_out: ' + str(self.Cs[C_out_idx]) + ' c_in: ' + str(self.Cs[C_in_idx]) + ' op type: ' + str(op_type_idx))
            # combined values with the same c_out dimension
            combined = sum(c_outs)
            if s0s[stride][C_out_idx] is None:
              # first call sets the value
              s0s[stride][C_out_idx] = combined
            else:
              s0s[stride][C_out_idx] += combined

      # downscale reduced input as next output
      s0s = [s0s[stride], [None] * self.C_size, [None] * self.C_size]

    # combine results
    # use SharpSepConv to match dimension of final linear layer
    # then add up all remaining outputs and pool the result
    out = self.global_pooling(sum(op(x) for op, x in zip(self.base, s0s[0]) if x is not None))
    # outs = []
    # print('len s0s[0]: ' + str(len(s0s[0])))
    # for i, op in enumerate(self.base):
    #   x = s0s[0][i]
    #   if x is not None:
    #     outs += [op()]
    # out = sum(outs)
    # out = self.global_pooling(out)
    logits = self.classifier(out.view(out.size(0),-1))
    # print('logits')
    #print("Optimal_path_forward", nx.algorithms.dag.dag_longest_path(self.G))
    #print("Top down greedy", self.gen_greedy_path(self.G,"top_down"))
    #print("Bottom up greedy",self.gen_greedy_path(self.G,"bottom_up"))
    return logits

  def gen_greedy_path(self, G, strategy="top_down"):
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
  def arch_weights(self, stride_idx):
    # ops are stored as layer, stride, cin, cout, num_layer_types
    # while weights are ordered stride_index, layer, cin, cout, num_layer_types
    # first exclude the stride_idx because we already know that
    view_shape = self.arch_weights_shape[1:]
    # print('arch_weights() view_shape self.weights_shape[1:]: ' + str(view_shape))
    # softmax of weights should occur once for each layer
    num_layers = self.arch_weights_shape[1]
    weights_softmax_view = self._arch_parameters[stride_idx].view(num_layers, -1)
    # apply softmax and convert to an indexable view
    weights = F.softmax(weights_softmax_view, dim=-1).view(view_shape)
    return weights

  def _loss(self, input_batch, target):
    logits = self(input_batch)
    return self._criterion(logits, target)

  def _initialize_alphas(self, genotype=None):

    if genotype is None or genotype[-1] == 'longest_path':
        init_alpha = 1e-3*torch.randn(self.arch_weights_shape)
    else:
        print("_initialize_alphas with preconfigured weights", genotype[0][0][0][0])
        init_alpha = []
        init_alpha.append(genotype[0])
        init_alpha.append(genotype[2])
        init_alpha = torch.from_numpy(np.array(init_alpha)).float()
    if torch.cuda.is_available():
      self._arch_parameters = Variable(init_alpha.cuda(), requires_grad=True)
    else:
      self._arch_parameters = Variable(init_alpha, requires_grad=True)

  def arch_parameters(self):
    ''' Get list of architecture parameters
    '''
    return [self._arch_parameters]

  def genotype(self, layout='raw_weights'):
    """
    layout options: raw_weights, longest_path, graph
    """
    if layout == 'raw_weights':
      # TODO(ahundt) switch from raw weights to a simpler representation for genotype?
      gene_normal = np.array(self.arch_weights(0).data.cpu().numpy()).tolist()
      gene_reduce = np.array(self.arch_weights(1).data.cpu().numpy()).tolist()
    elif layout == 'longest_path':
      # TODO(ahundt) make into a list of the layer strings to be included.
      gene_normal = nx.algorithms.dag.dag_longest_path(self.G)
      gene_reduce = []
    elif layout == 'graph':
      data = json_graph.node_link_data(self.G)
      gene_normal = [json.dumps(data)]
      gene_reduce = []
    else:
      raise ValueError('unsupported layout: ' + str(layout))

    genotype = Genotype(
      normal=gene_normal, normal_concat=[],
      reduce=gene_reduce, reduce_concat=[],
      layout=layout
    )
    return genotype
