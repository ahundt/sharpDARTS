import torch
import torch.nn as nn
import torch.nn.functional as F
import operations
from operations import ReLUConvBN
from operations import ConvBNReLU
from operations import FactorizedReduce
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import REDUCE_PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C_in, C_out, stride, primitives=None, op_dict=None):
    """ Perform a mixed forward pass incorporating multiple primitive operations like conv, max pool, etc.

    # Arguments

      primitives: the list of strings defining the operations to choose from.
      op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict.
    """
    super(MixedOp, self).__init__()
    self._stride = stride
    self._ops = nn.ModuleList()
    if primitives is None:
          primitives = PRIMITIVES
    if op_dict is None:
          op_dict = operations.OPS
    # print('-------------------- init')
    for primitive in primitives:
      op = op_dict[primitive](C_in, C_out, stride, False)
      if 'pool' in primitive:
        # this batchnorm might be added because max pooling will essentially
        # give itself extra high weight, since it takes the largest of its inputs
        op = nn.Sequential(op, nn.BatchNorm2d(C_out, affine=False))
      # print('primitive: ' + str(primitive) + ' cin: ' + str(C_in) + ' cout: ' + str(C_out) + ' stride: ' + str(stride))
      self._ops.append(op)

  def forward(self, x, weights):
    result = 0
    # print('-------------------- forward')
    # print('weights shape: ' + str(len(weights)) + ' ops shape: ' + str(len(self._ops)))
    # for i, (w, op) in enumerate(zip(weights, self._ops)):
    #   # print('w shape: ' + str(w.shape) + ' op type: ' + str(type(op)) + ' i: ' + str(i) + ' PRIMITIVES[i]: ' + str(PRIMITIVES[i]) + 'x size: ' + str(x.size()) + ' stride: ' + str(self._stride))
    #   op_out = op(x)
    #   # print('op_out size: ' + str(op_out.size()))
    #   result += w * op_out
    # return result
    # apply all ops with intensity corresponding to their weight
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class MixedAux(nn.Module):

  def __init__(self, num_classes, weights_are_parameters=True):
    """ Perform a mixed Auxiliary Layer incorporating the output of multiple cells.

    # Arguments

      num_classes: number of outputs for each linear layer
      weights_are_parameters: weights of the various aux layers are network parameters
    """
    super(MixedAux, self).__init__()
    self._num_classes = num_classes
    self._ops = nn.ModuleList()
    self._weights_are_parameters = weights_are_parameters
    self.global_pooling = nn.AdaptiveMaxPool2d(1)

  def add_aux(self, C_prev):
    op = nn.Linear(C_prev, self._num_classes)
    self._ops.append(op)

  def initialize_alphas(self):
      self.alphas = 1e-3 * torch.randn(len(self._ops), requires_grad=True).cuda()
      if self._weights_are_parameters:
        # in simpler training modes the weights are just regular parameters
        self.alphas = torch.nn.Parameter(self.alphas)

  def get_weights(self):
    return F.softmax(self.alphas, dim=-1)

  def forward(self, xs):
    result = 0
    weights = self.get_weights()
    # print('-------------------- forward')
    # print('weights shape: ' + str(len(weights)) + ' ops shape: ' + str(len(self._ops)))
    # for i, (w, op) in enumerate(zip(weights, self._ops)):
    #   # print('w shape: ' + str(w.shape) + ' op type: ' + str(type(op)) + ' i: ' + str(i) + ' PRIMITIVES[i]: ' + str(PRIMITIVES[i]) + 'x size: ' + str(x.size()) + ' stride: ' + str(self._stride))
    #   op_out = op(x)
    #   # print('op_out size: ' + str(op_out.size()))
    #   result += w * op_out
    # return result
    # apply all ops with intensity corresponding to their weight
    logits = []
    for w, op, x in zip(weights, self._ops, xs):
      out = self.global_pooling(x)
      logits += [w * op(out.view(out.size(0), -1))]
    return sum(logits)

  def genotype(self):
    return [float(w) for w in self.get_weights()]



class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev,
               alphas, primitives=None, op_dict=None):
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
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier
    self._alphas = alphas

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        # print('i: ' + str(i) + ' j: ' + str(j) + ' stride: ' + str(stride))
        op = MixedOp(C, C, stride=stride, primitives=primitives, op_dict=op_dict)
        self._ops.append(op)

  def get_weights(self):
    return F.softmax(self.alphas, dim=-1)

  def forward(self, s0, s1):
    weights = self.get_weights()
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
  """Network for architecture search.
  """

  def __init__(self, C, num_classes, layers, criterion, in_channels=3, steps=4,
               multiplier=4, stem_multiplier=3, reduce_spacing=None, primitives=None,
               reduce_primitives=None, op_dict=None, weights_are_parameters=False,
               mixed_aux=True, start_cell_weights=True, end_cell_weights=True):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._in_channels = in_channels
    self._reduce_spacing = reduce_spacing
    if primitives is None:
          primitives = PRIMITIVES
    self._primitives = primitives
    if reduce_primitives is None:
          reduce_primitives = REDUCE_PRIMITIVES
    self._reduce_primitives = reduce_primitives
    self._num_primitives = len(primitives)
    self._num_reduce_primitives = len(reduce_primitives)
    self._weights_are_parameters = weights_are_parameters
    self._alphas_start = None
    if start_cell_weights:
          self._alphas_start = True

    self._alphas_end = None
    if end_cell_weights:
          self._alphas_end = True

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    if mixed_aux:
      self.auxs = MixedAux(num_classes, weights_are_parameters=weights_are_parameters)
    else:
      self.auxs = None

    self._initialize_alphas()

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    # create each cell
    for i in range(layers):
      if ((reduce_spacing is None and i in [layers//3, 2*layers//3]) or
          (reduce_spacing is not None and ((i + 1) % reduce_spacing == 0))):
        C_curr *= 2
        reduction = True
        primitives = self._reduce_primitives
      else:
        reduction = False
        primitives = self._primitives

      # alphas are the weights to choose an architecture and
      # there may be different architectures for:
      # 1. the initial start cell,
      # 2. normal cells in the middle
      # 3. resolution reduction cells in the middle
      # 4. the final end cell
      if i == 0 and self._alphas_start is not None:
        alphas = self._alphas_start
        print('>>>>>>> network init cell i: ' + str(i) + ' with start alphas')
      elif i == layers - 1 and self._alphas_end is not None:
        alphas = self.alphas_end
        print('>>>>>>> network init cell i: ' + str(i) + ' with end alphas')
      elif reduction:
        alphas = self.alphas_reduce
        print('>>>>>>> network init cell i: ' + str(i) + ' with reduce alphas')
      else:
        alphas = self.alphas_normal
        print('>>>>>>> network init cell i: ' + str(i) + ' with normal alphas')
      # print('>>>>>>> network init cell i: ' + str(i))
      cell = Cell(steps=steps, multiplier=multiplier, C_prev_prev=C_prev_prev,
                  C_prev=C_prev, C=C_curr, reduction=reduction, reduction_prev=reduction_prev,
                  primitives=primitives, op_dict=op_dict, alphas=alphas)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr
      # auxiliary networks for final decision making accelerate training
      if self.auxs is not None:
        self.auxs.add_aux(C_prev)

    if self.auxs is None:
      self.global_pooling = nn.AdaptiveMaxPool2d(1)
      # self.global_pooling = nn.AdaptiveAvgPool2d(1)
      self.classifier = nn.Linear(C_prev, num_classes)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._in_channels).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def copy_arch_parameters(self, model_old):
    """ Copy Architecture Parameters from another model
    """
    for x, y in zip(self.arch_parameters(), model_old.arch_parameters()):
        x.data.copy_(y.data)

  def forward(self, input_batch):
    s0 = s1 = self.stem(input_batch)
    s1s = []
    for i, cell in enumerate(self.cells):
      # print('\nnormal i: ' + str(i) + ' len weights: ' + str(len(weights)))
      # print('<<<<<<< network forward cell i: ' + str(i))
      s0, s1 = s1, cell(s0, s1)
      # get the outputs for multiple aux networks
      if self.auxs is not None:
        s1s += [s1]

    if self.auxs is not None:
      # combine the result of all aux networks
      logits = self.auxs(s1s)
    else:
      out = self.global_pooling(s1)
      logits = self.classifier(out.view(out.size(0),-1))

    return logits

  def _loss(self, input_batch, target):
    logits = self(input_batch)
    return self._criterion(logits, target)

  def set_criterion(self, criterion):
    self._criterion = criterion

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = self._num_primitives
    num_reduce_ops = self._num_reduce_primitives
    self._arch_parameters = []
    # print('\nk: ' + str(k) + ' num_ops: ' + str(num_ops) + ' num_reduce_ops: ' + str(num_reduce_ops))

    # the quantity of alphas is the number of primitives * k
    # and k is based on the number of steps
    # TODO(ahundt) attempted fix by removing more efficient alphas when no reductions are used to try fixing crash.
    if self.alphas_start is not None:
      self.alphas_start = 1e-3 * torch.randn(k, num_ops, requires_grad=True).cuda()
      if self._weights_are_parameters:
        # in simpler training modes the weights are just regular parameters
        self.alphas_start = torch.nn.Parameter(self.alphas_start)
      self._arch_parameters += [self.alphas_start]

    if self._reduce_spacing is None or self._reduce_spacing != 1:
      # reduce spacing of 1 means there won't be any normal layers
      self.alphas_normal = 1e-3 * torch.randn(k, num_ops, requires_grad=True).cuda()
      self.alphas_reduce = 1e-3 * torch.randn(k, num_reduce_ops, requires_grad=True).cuda()

      if self._weights_are_parameters:
        # in simpler training modes the weights are just regular parameters
        self.alphas_normal = torch.nn.Parameter(self.alphas_normal)
        self.alphas_reduce = torch.nn.Parameter(self.alphas_reduce)
      self._arch_parameters += [
        self.alphas_normal,
        self.alphas_reduce,
      ]
    else:
      self.alphas_normal = None
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_reduce_ops).cuda(), requires_grad=True)

      if self._weights_are_parameters:
        # in simpler training modes the weights are just regular parameters
        self.alphas_reduce = torch.nn.Parameter(self.alphas_reduce)
      self._arch_parameters = [self.alphas_reduce]

    if self.alphas_end is not None:
      self.alphas_end = 1e-3 * torch.randn(k, num_ops, requires_grad=True).cuda()
      if self._weights_are_parameters:
        # in simpler training modes the weights are just regular parameters
        self.alphas_end = torch.nn.Parameter(self.alphas_end)
        self._arch_parameters += [self.alphas_start]

    if self.auxs is not None:
          self.auxs.initialize_alphas()
          self._arch_parameters += [self.auxs.alphas]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights, primitives):
      """Extract the names of the layers to use from the weights.
      """
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != primitives.index('none')))[:2]
        for j in edges:
          # (key, weight, primitive)
          # (  0,      1,         2)
          kwp_best = (None, None, None)
          kwp_2nd = (None, None, None)

          for k in range(len(W[j])):
            if k != primitives.index('none'):
              w = W[j][k]
              if kwp_best[0] is None or W[j][k] > kwp_best[1]:
                kwp_2nd = kwp_best
                kwp_best = (k, w, primitives[k])
          gene.append((kwp_best[2], j))
          if 'pool' in kwp_best[2]:
            # trying to analyze if maxpool is actually so powerful or if there is usually a close second
            # indicating there is an architecture problem that could be solved
            print('\npool is best showing second (key, weight, primitive), 1st:' + str(kwp_best) +
                  ' 2nd: ' + str(kwp_2nd) + ' j: ' + str(j))
        start = end
        n += 1
      return gene

    def _set_gene_concat(alphas, primitives):
      """ Set empty variables if needed, or extract gene layer type + index using _parse call.
      """
      if alphas is None:
        gene = []
        concat = []
      else:
        gene = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy(), primitives)
        concat = range(2+self._steps-self._multiplier, self._steps+2)
      return gene, concat

    gene_reduce, reduce_concat = _set_gene_concat(self.alphas_reduce, self._reduce_primitives)
    gene_normal, normal_concat = _set_gene_concat(self.alphas_normal, self._primitives)
    gene_start, start_concat = _set_gene_concat(self.alphas_start, self._primitives)
    gene_end, end_concat = _set_gene_concat(self.alphas_end, self._primitives)

    aux = []
    # TODO(ahundt) determine criteria for final decision on aux networks
    # get weights assigned to auxiliary networks
    if self.auxs is not None:
        aux = self.auxs.genotype()

    genotype = Genotype(
      normal=gene_normal, normal_concat=normal_concat,
      reduce=gene_reduce, reduce_concat=reduce_concat,
      start=gene_start, start_concat=start_concat,
      end=gene_end, end_concat=end_concat,
      aux=aux,
    )
    return genotype

