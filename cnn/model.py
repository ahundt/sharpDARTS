import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# from . import operations
# from . import genotypes
# from .operations import ReLUConvBN
# from .operations import ConvBNReLU
# from .operations import FactorizedReduce
# from .operations import Identity
from torch.autograd import Variable
# from .utils import drop_path
# from .model_search import MixedAux
import operations
import genotypes
from operations import FactorizedReduce
from operations import Identity
from operations import ReLUConvBN
from operations import SepConv
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype_sequence, concat_sequence, C_prev_prev, C_prev, C, reduction, reduction_prev,
               op_dict=None, separate_reduce_cell=True, C_mid=None):
    """Create a final cell with a single architecture.

    The Cell class in model_search.py is the equivalent for searching multiple architectures.

    # Arguments

      op_dict: The dictionary of possible operation creation functions.
        All primitive name strings defined in the genotype must be in the op_dict.
    """
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    self.reduction = reduction
    if op_dict is None:
      op_dict = operations.OPS
    # _op_dict are op_dict available for use,
    # _ops is the actual sequence of op_dict being utilized in this case
    self._op_dict = op_dict

    if reduction_prev is None:
      self.preprocess0 = operations.Identity()
    elif reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, stride=2)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    op_names, indices = zip(*genotype_sequence)
    self._compile(C, op_names, indices, concat_sequence, reduction, C_mid)

  def _compile(self, C, op_names, indices, concat, reduction, C_mid):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = self._op_dict[name](C, C, stride, True, C_mid)
      # op = self._op_dict[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob=0.):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE(Hanxaio Liu): This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=3, reduce_spacing=None,
               mixed_aux=False, op_dict=None, C_mid=None, stem_multiplier=3):
    """
    # Arguments

        C: Initial number of output channels.
        in_channels: initial number of input channels
        layers: The number of cells to create.
        reduce_spacing: number of layers of cells between reduction cells,
            default of None is at 1/3 and 2/3 of the total number of layers.
            1 means all cells are reduction. 2 means the first layer is
            normal then the second
        op_dict: The dictionary of possible operation creation functions.
            All primitive name strings defined in the genotype must be in the op_dict.
    """
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels
    self.drop_path_prob = 0.

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    if mixed_aux:
      self.auxs = MixedAux(num_classes, weights_are_parameters=True)
    else:
      self.auxs = None

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if ((reduce_spacing is None and i in [layers//3, 2*layers//3]) or
          (reduce_spacing is not None and ((i + 1) % reduce_spacing == 0))):
        C_curr *= 2
        reduction = True
        cell = Cell(genotype.reduce, genotype.reduce_concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, op_dict=op_dict, C_mid=C_mid)
      else:
        reduction = False
        cell = Cell(genotype.normal, genotype.normal_concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, op_dict=op_dict, C_mid=C_mid)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if self.auxs is not None:
        self.auxs.add_aux(C_prev)
      elif i == 2*layers//3:
        C_to_auxiliary = C_prev

    if self.auxs is None:
      if auxiliary:
        self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
      # self.global_pooling = nn.AdaptiveMaxPool2d(1)
      self.global_pooling = nn.AdaptiveAvgPool2d(1)
      self.classifier = nn.Linear(C_prev, num_classes)
    else:
      # init params to prioritize auxiliary decision making networks
      self.auxs.build()

  def forward(self, input_batch):
    logits_aux = None
    s0 = s1 = self.stem(input_batch)
    s1s = []
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if self.auxs is not None:
        # print('network forward i: ' + str(i) + ' s1 shape: ' + str(s1.shape))
        s1s += [s1]
      elif i == 2 * self._layers // 3 and self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)

    if self.auxs is not None:
      # combine the result of all aux networks
      # print('calling auxs, s1s len: ' + str(len(s1s)))
      logits = self.auxs(s1s)
    else:
      out = self.global_pooling(s1)
      logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=3, reduce_spacing=None,
               mixed_aux=False, op_dict=None, C_mid=None, stem_multiplier=3):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels
    self.drop_path_prob = 0.

    self.stem0 = nn.Sequential(
      nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
        cell = Cell(genotype.reduce, genotype.reduce_concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, op_dict=op_dict, C_mid=C_mid)
      else:
        reduction = False
        cell = Cell(genotype.normal, genotype.reduce_concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, op_dict=op_dict, C_mid=C_mid)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, batch_input):
    logits_aux = None
    s0 = self.stem0(batch_input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux
