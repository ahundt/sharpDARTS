import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import networkx as nx


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
    #   print('w shape: ' + str(w.shape) + ' op type: ' + str(type(op)) + ' i: ' + str(i) + ' PRIMITIVES[i]: ' + str(PRIMITIVES[i]) + 'x size: ' + str(x.size()) + ' stride: ' + str(self._stride))
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
      self.preprocess0 = SepConv(C_prev_prev, C, stride=2, affine=False)
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
    num_ops = len(PRIMITIVES)

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

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


class MultiChannelNetwork(nn.Module):

  def __init__(self, C=32, num_classes=10, layers=6, criterion=None, steps=5, multiplier=4, stem_multiplier=3,
               in_channels=3, final_linear_filters=768, always_apply_ops=False, visualization=False,
               weighting_algorithm=None):
    """ C is the mimimum number of channels. Layers is how many output scaling factors and layers should be in the network.
    """
    super(MultiChannelNetwork, self).__init__()
    self._C = C
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
    self.op_types = [operations.SharpSepConv, operations.ResizablePool]
    self.stem = nn.ModuleList()
    self.G = nx.DiGraph()
    for i, c in enumerate(self.Cs):
      s = nn.Sequential(
        nn.Conv2d(int(in_channels), int(c), 3, padding=1, bias=False),
        nn.BatchNorm2d(c)
      )
      self.G.add_node("Conv3x3_"+str(i))
      self.G.add_node("BatchNorm_"+str(i))
      self.G.add_edge("Conv3x3_"+str(i), "BatchNorm_"+str(i))
      self.stem.append(s)
    for layer_idx in range(self._layers):
        for C_out_idx in range(self.C_size):
            out_node = 'layer_'+str(layer_idx)+' add '+'c_out'+str(self.Cs[C_out_idx])
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
            out_node = 'layer_'+str(layer_idx)+' add '+'c_out'+str(self.Cs[C_out_idx])
            type_modules = nn.ModuleList()
            for OpType in self.op_types:
              cin = C_in[C_in_idx][C_out_idx]
              cout = C_out[C_in_idx][C_out_idx]
              # print('cin: ' + str(cin) + ' cout: ' + str(cout))
              name = 'layer_' + str(layer_idx) + '_stride_' + str(stride_idx+1) + '_c_in_' + str(self.Cs[C_in_idx]) + '_c_out_' + str(self.Cs[C_out_idx]) + '_op_type_' + str(OpType.__name__)
              self.G.add_node(name)
              if layer_idx == 0:
                self.G.add_edge("BatchNorm_"+str(C_in_idx), name)
              else:
                self.G.add_edge('layer_' + str(layer_idx-1)+' add ' + 'c_out'+str(self.Cs[C_in_idx]), name)
              self.G.add_edge(name, out_node)
              op = OpType(int(cin), int(cout), kernel_size=3, stride=int(stride_idx + 1))
              type_modules.append(op)
            out_modules.append(type_modules)
          in_modules.append(out_modules)
        # op grid is stride_modules
        stride_modules.append(in_modules)
      self.op_grid.append(stride_modules)

    self.base = nn.ModuleList()
    self.G.add_node("Add-SharpSep")
    for C_out_idx in range(self.C_size):
        self.G.add_edge('layer_'+str(self._layers-1)+' add '+'c_out'+str(self.Cs[C_out_idx]), "Add-SharpSep")
    for c in self.Cs:
      self.G.add_node("SharpSepConv" + str(c))
      self.G.add_edge("SharpSepConv" + str(c), "Add-SharpSep")
      self.base.append(operations.SharpSepConv(int(c), int(final_linear_filters), 3))
    # TODO(ahundt) there should be one more layer of normal convolutions to set the final linear layer size
    # C_in will be defined by the previous layer's c_out
    self.arch_weights_shape = [len(self.strides), self._layers, self.C_size, self.C_size, len(self.op_types)]
    # number of weights total
    self.weight_count = np.prod(self.arch_weights_shape)
    # number of weights in a softmax call
    self.softmax_weight_count = np.prod(self.arch_weights_shape[2:])
    # minimum score for a layer to continue being trained
    self.min_score = float(1 / (self.softmax_weight_count * self.softmax_weight_count))

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(final_linear_filters, num_classes)
    self.G.add_node("global_pooling")
    self.G.add_edge("Add-SharpSep", "global_pooling")
    self.G.add_node("Linear")
    self.G.add_edge("global_pooling", "Linear")
    print("Nodes in graph")
    print(self.G.nodes())
    print("Edges in graph")
    print(self.G.edges())
    print("Saving graph...")
    nx.write_gpickle(self.G, "network_test.graph")

    if not self._visualization:
      self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input_batch):
    # [in, normal_out, reduce_out]
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
    for layer in range(self._layers):
      # layer is how many times we've called everything, i.e. the number of "layers"
      # this is different from the number of layer types which is len([SharpSepConv, ResizablePool]) == 2
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
          out_node = 'layer_'+str(layer)+' add '+'c_out'+str(C_out)
          c_outs = []
          for C_in_idx, C_in in enumerate(self.Cs):
            for op_type_idx, OpType in enumerate(self.op_types):
              # get the specific weight for this op
              name = 'layer_' + str(layer) + '_stride_' + str(stride_idx+1) + '_c_in_' + str(C_in) + '_c_out_' + str(C_out) + '_op_type_' + str(OpType.__name__)
              if not self._visualization:
                w = weight_views[stride_idx][layer, C_in_idx, C_out_idx, op_type_idx]
              # self.G.add_edge(name, out_node, {weight: w})
              self.G[name][out_node]["weight"] = w
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
                      x = w * self.op_grid[layer][stride_idx][C_in_idx][C_out_idx][op_type_idx](s)
                    elif self._weighting_algorithm == 'max_w':
                      x = (1. - max_w + w) * self.op_grid[layer][stride_idx][C_in_idx][C_out_idx][op_type_idx](s)
                    else:
                      raise ValueError(
                        'MultiChannelNetwork.forward(): Unsupported weighting algorithm: ' +
                        str(self._weighting_algorithm) + ' try "scalar" or "max_w"')
                  else:
                    # doing visualization, skip the weights
                    x = self.op_grid[layer][stride_idx][C_in_idx][C_out_idx][op_type_idx](s)
                  c_outs += [x]
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
    return logits

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

  def _initialize_alphas(self):
    if torch.cuda.is_available():
      self._arch_parameters = Variable(1e-3*torch.randn(self.arch_weights_shape).cuda(), requires_grad=True)
    else:
      self._arch_parameters = Variable(1e-3*torch.randn(self.arch_weights_shape), requires_grad=True)

  def arch_parameters(self):
    ''' Get list of architecture parameters
    '''
    return [self._arch_parameters]

  def genotype(self):
    # TODO(ahundt) switch from raw weights to a simpler representation for genotype?
    gene_normal = np.array(self.arch_weights(0).data.cpu().numpy()).tolist()
    gene_reduce = np.array(self.arch_weights(1).data.cpu().numpy()).tolist()

    genotype = Genotype(
      normal=gene_normal, normal_concat=[],
      reduce=gene_reduce, reduce_concat=[]
    )
    return genotype