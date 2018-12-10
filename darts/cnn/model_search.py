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
    print('-------------------- init')
    for primitive in primitives:
      op = op_dict[primitive](C_in, C_out, stride, False)
      if 'pool' in primitive:
        # this batchnorm might be added because max pooling will essentially
        # give itself extra high weight, since it takes the largest of its inputs
        op = nn.Sequential(op, nn.BatchNorm2d(C_out, affine=False))
      print('primitive: ' + str(primitive) + ' cin: ' + str(C_in) + ' cout: ' + str(C_out) + ' stride: ' + str(stride))
      self._ops.append(op)

  def forward(self, x, weights):
    result = 0
    print('-------------------- forward')
    print('weights shape: ' + str(len(weights)) + ' ops shape: ' + str(len(self._ops)))
    for i, (w, op) in enumerate(zip(weights, self._ops)):
      print('w shape: ' + str(w.shape) + ' op type: ' + str(type(op)) + ' i: ' + str(i) + ' PRIMITIVES[i]: ' + str(PRIMITIVES[i]) + 'x size: ' + str(x.size()) + ' stride: ' + str(self._stride))
      op_out = op(x)
      print('op_out size: ' + str(op_out.size()))
      result += w * op_out
    return result
    # apply all ops with intensity corresponding to their weight
    return sum(w * op(x) for w, op in zip(weights, self._ops))



class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev,
               primitives=None, op_dict=None):
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

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        # print('i: ' + str(i) + ' j: ' + str(j) + ' stride: ' + str(stride))
        op = MixedOp(C, C, stride=stride, primitives=primitives, op_dict=op_dict)
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
  """Network for architecture search.
  """

  def __init__(self, C, num_classes, layers, criterion, in_channels=3, steps=4,
               multiplier=4, stem_multiplier=3, reduce_spacing=None, primitives=None,
               reduce_primitives=None, op_dict=None):
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

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

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
      print('>>>>>>> network init cell i: ' + str(i))
      cell = Cell(steps=steps, multiplier=multiplier, C_prev_prev=C_prev_prev,
                  C_prev=C_prev, C=C_curr, reduction=reduction, reduction_prev=reduction_prev,
                  primitives=primitives, op_dict=op_dict)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveMaxPool2d(1)
    # self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._in_channels).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input_batch):
    s0 = s1 = self.stem(input_batch)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        # print('\nreduction i: ' + str(i) + ' len weights: ' + str(len(weights)))
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        # print('\nnormal i: ' + str(i) + ' len weights: ' + str(len(weights)))
      print('<<<<<<< network forward cell i: ' + str(i))
      s0, s1 = s1, cell(s0, s1, weights)
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
    # print('\nk: ' + str(k) + ' num_ops: ' + str(num_ops) + ' num_reduce_ops: ' + str(num_reduce_ops))

    # the quantity of alphas is the number of primitives * k
    # and k is based on the number of steps
    # TODO(ahundt) attempted fix by removing more efficient alphas when no reductions are used to try fixing crash.
    hack_around_crash = True
    if hack_around_crash or self._reduce_spacing != 1:
      # reduce spacing of 1 means there won't be any normal layers
      self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_reduce_ops).cuda(), requires_grad=True)
      self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
      ]
    else:
      self.alphas_normal = None
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_reduce_ops).cuda(), requires_grad=True)
      self._arch_parameters = [self.alphas_reduce]

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
          k_best = None
          for k in range(len(W[j])):
            if k != primitives.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((primitives[k_best], j))
        start = end
        n += 1
      return gene

    # Determine the final selection of layers for the network
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), self._reduce_primitives)
    # The concatenations to apply are pre-determined
    reduce_concat = range(2+self._steps-self._multiplier, self._steps+2)

    if self.alphas_normal is not None:
      gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), self._primitives)
      normal_concat = reduce_concat
    else:
      # all reduction networks don't have any normal cells
      gene_normal = []
      normal_concat = []

    genotype = Genotype(
      normal=gene_normal, normal_concat=normal_concat,
      reduce=gene_reduce, reduce_concat=reduce_concat
    )
    return genotype

