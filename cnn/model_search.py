import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self._stride = stride
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, C, stride, False)
      # op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

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
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
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
        op = MixedOp(C, stride)
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

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
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
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
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
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

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

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(MultiChannelNetwork, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    self.normal_index = 0
    self.reduce_index = 1
    self.layer_types = 2
    self.strides = np.array([self.normal_index, self.reduce_index])
    self.C_start = 5
    self.C_end = 10
    self.Cs = np.array(np.exp2(np.arange(self.C_start,self.C_end)), dtype='int')
    # $ print(Cs)
    # [ 32.  64. 128. 256. 512.]
    self.C_size = len(self.Cs)
    C_in, C_out = np.array(np.meshgrid(self.Cs,self.Cs, indexing='ij'), dtype='int')
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
    self.op_types = [operations.SepConv, operations.ResizablePool]
    self.stem = nn.ModuleList()
    for c in self.Cs:
      s = nn.Sequential(
        nn.Conv2d(3, c, 3, padding=1, bias=False),
        nn.BatchNorm2d(c)
      )
      self.stem.append(s)

    self.op_grid = nn.ModuleList()
    for stride_idx in self.strides:
      in_modules = nn.ModuleList()
      for C_in_idx in range(self.C_size):
        out_modules = nn.ModuleList()
        for C_out_idx in range(self.C_size):
          type_modules = nn.ModuleList()
          for OpType in self.op_types:
            cin = C_in[C_in_idx][C_out_idx]
            cout = C_out[C_in_idx][C_out_idx]
            # print('cin: ' + str(cin) + ' cout: ' + str(cout))
            op = OpType(cin, cout, kernel_size=3, stride=stride_idx + 1)
            type_modules.append(op)
          out_modules.append(type_modules)
        in_modules.append(out_modules)
      # op grid is stride_modules
      self.op_grid.append(in_modules)

    # C_in will be defined by the previous layer's c_out
    self.arch_weights_shape = [len(self.strides), layers, self.C_size, self.C_size, len(self.op_types)]
    # number of weights total
    self.weight_count = np.prod(self.arch_weights_shape)
    # number of weights in a softmax call
    self.softmax_weight_count = np.prod(self.arch_weights_shape[2:])
    # minimum score for a layer to continue being trained
    self.min_score = 1 / (self.softmax_weight_count * self.softmax_weight_count)

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(np.max(self.Cs), num_classes)

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
    for stride_idx in self.strides:
      # ops are stored as layer, stride, cin, cout, num_layer_types
      # while weights are ordered stride_index, layer, cout, num_layer_types
      # first exclude the stride_idx because we already know that
      weight_views += [self.weights(stride_idx)]
    # Duplicate s0s to account for 2 different strides
    # s0s += [[]]
    # s1s = [None] * layers + 1
    for layer in range(self._layers):
      # layer is how many times we've called everything, i.e. the number of "layers"
      # this is different from the number of layer types which is len([SepConv, ResizablePool]) == 2
      for stride_idx in self.strides:
        for C_out_idx, C_out in enumerate(self.Cs):
          # take all the layers with the same output so we can sum them
          c_outs = []
          for C_in_idx, C_in in enumerate(self.Cs):
            for op_type_idx in range(len(self.op_types)):
              # get the specific weight for this op
              w = weight_views[stride_idx][layer, C_in_idx, C_out_idx, op_type_idx]
              # print('w weight_views[stride_idx][layer, C_in_idx, C_out_idx, op_type_idx]: ' + str(w))
              # apply the operation then weight, equivalent to
              # w * op(input_feature_map)
              if w > self.min_score:
                # only apply an op if weight score isn't too low: w > 1/(N*N)
                x = w * self.op_grid[C_in_idx][C_out_idx][op_type_idx](s0s[stride_idx][C_in_idx])
              c_outs += [x]
          # combined values with the same c_out dimension
          combined = sum(c_outs)
          s_idx = 1 + stride_idx
          if s0s[s_idx][C_out_idx] is None:
            # first call sets the value
            s0s[s_idx][C_out_idx] = combined
          else:
            s0s[s_idx][C_out_idx] += combined

      # downscale reduced input as next output
      s0s = [s0s[2], [None] * self.C_size, [None] * self.C_size]

    out = s0s[0][-1]
    out = self.global_pooling(out)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def weights(self, stride_idx):
    # ops are stored as layer, stride, cin, cout, num_layer_types
    # while weights are ordered stride_index, layer, cout, num_layer_types
    # first exclude the stride_idx because we already know that
    view_shape = self.arch_weights_shape[1:]
    # print('weights() view_shape self.weights_shape[1:]: ' + str(view_shape))
    # softmax of weights should occur once for each layer
    num_layers = self.arch_weights_shape[1]
    weights_softmax_view = self._arch_parameters[stride_idx].view(num_layers, -1)
    # apply softmax and convert to an indexable view
    weights = F.softmax(weights_softmax_view, dim=-1).view(view_shape)
    return weights

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    self._arch_parameters = Variable(1e-3*torch.randn(self.arch_weights_shape).cuda(), requires_grad=True)

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    # TODO(ahundt) switch from raw weights to a simpler representation for genotype?
    gene_normal = [self.weights(0).data.cpu().numpy()]
    gene_reduce = [self.weights(1).data.cpu().numpy()]

    genotype = Genotype(
      normal=gene_normal, normal_concat=[],
      reduce=gene_reduce, reduce_concat=[]
    )
    return genotype