import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from genotypes import PRIMITIVES, MULTICHANNELNET_PRIMITIVES
from operations import *
import operations
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


class NoisyLinear(nn.Module):
  """ Factorised NoisyLinear layer with bias
  Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning https://arxiv.org/abs/1710.02298
  Code Source: https://github.com/Kaixhin/Rainbow
  """
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  """
  Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning https://arxiv.org/abs/1710.02298
  Code Source: https://github.com/Kaixhin/Rainbow
  """
  def __init__(self, args, action_space):
    super(DQNAS, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)
    q = self.q_score(x, log)  # Probabilities with action over second dimension
    return q

  def q_score(self, x, log):
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class RainbowDenseBlock(nn.Module):
  """Decision making layers of the Rainbow reinforcement learning algorithm.

  Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning https://arxiv.org/abs/1710.02298
  Code Source: https://github.com/Kaixhin/Rainbow
  """
  def __init__(self, c_in, action_space, hidden_size=512, atoms=51, noisy_std=0.1):
    """
    # Arguments

      c_in: number of channels in.
      action_space: number of discrete possible actions, like controller buttons.
      atoms: Discretised size of value distribution.
      hidden_size: Network hidden size
    """
    super().__init__()
    self.c_in = c_in
    self.atoms = atoms
    self.action_space = action_space
    self.fc_h_v = NoisyLinear(self.c_in, hidden_size, std_init=noisy_std)
    self.fc_h_a = NoisyLinear(self.c_in, hidden_size, std_init=noisy_std)
    self.fc_z_v = NoisyLinear(hidden_size, self.atoms, std_init=noisy_std)
    self.fc_z_a = NoisyLinear(hidden_size, action_space * self.atoms, std_init=noisy_std)

  def forward(self, x, log=False):
    x = x.view(-1, self.c_in)
    q = self.q_score(x, log)  # Probabilities with action over second dimension
    return q

  def q_score(self, x, log=False):
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()



class DQNAS(nn.Module):

  def __init__(self, C=36, num_classes=10, layers=4, auxiliary=False, genotype=None, in_channels=3, reduce_spacing=None, noisy_std=0.1, drop_path_prob=0.0):
    """
        # Arguments
        C: Initial number of output channels.
        in_channels: Initial number of input channels
        layers: The number of cells to create.
        auxiliary: Train a smaller auxiliary network partway down for "deep supervision" see NAS paper for details.
        in_channels: The number of channels for input data, for example rgb images have 3 input channels.
        reduce_spacing: number of layers of cells between reduction cells,
            default of None is at 1/3 and 2/3 of the total number of layers.
            1 means all cells are reduction. 2 means the first layer is
            normal then the second
        noisy_std: Initial standard deviation of noisy linear layers
    """
    super(DQNAS, self).__init__()
    if genotype is None:
          genotype = genotypes.DARTS_V2
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels
    self._noisy_std = noisy_std
    self._drop_path_prob = drop_path_prob

    C_prev_prev, C_prev = self.nas_build(C, in_channels, layers, reduce_spacing, genotype, auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # self.classifier = nn.Linear(C_prev, num_classes)
    self.classifier = RainbowDenseBlock(C_prev, num_classes)

  def nas_build(self, C, in_channels, layers, reduce_spacing, genotype, auxiliary, num_classes):
    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if ((reduce_spacing is None and i in [layers//3, 2*layers//3]) or
          (reduce_spacing is not None and ((i + 1) % reduce_spacing == 0))):
        C_curr *= 2
        reduction = True
        cell = Cell(genotype.reduce, genotype.reduce_concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      else:
        reduction = False
        if i == 0 and genotype.start:
          # start cell is nonempty
          sequence = genotype.start
          concat = genotype.start_concat
        elif i == layers - 1 and genotype.end:
          # end cell is nonempty
          sequence = genotype.end
          concat = genotype.end_concat
        else:
          # we are on a normal cell
          sequence = genotype.normal
          concat = genotype.normal_concat
        cell = Cell(sequence, concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

    return C_prev_prev, C_prev

  def forward(self, input, log=False):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self._drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1), log=log)
    if self._auxiliary:
      return logits, logits_aux
    else:
      #TODO(ahundt) previously a tuple was always returned and this if statement wasn't here before. check for compatibility with other DARTS code
      return logits

  def reset_noise(self):
        self.classifier.reset_noise()

class MultiChannelNetworkModel(nn.Module):
  """
    This class is used to initialize a sub-graph or linear model found after running a MultiChannelNet search (refer to MultiChannelNetwork 
    in model_search.py) and using graph operations found in multichannelnet_graph_operations.py.
    To see an example of how final model architectures are stored, refer to genotypes.py
    For more information about MultiChannelNet (Differentiable Grid Search) refer to section 4 of the paper https://arxiv.org/abs/1903.09900
  """

  def __init__(self, C=32, num_classes=10, layers=6, criterion=None, steps=5, multiplier=4, stem_multiplier=3,
               in_channels=3, final_linear_filters=768, always_apply_ops=False, visualization=False, primitives=None,
               op_dict=None, weighting_algorithm=None, genotype=None, simple_path=True):
    """ C is the mimimum number of channels. Layers is how many output scaling factors and layers should be in the network.
        op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict. (Refer to operations.py and genotypes.py for all the primitives available)
        genotype is used to get the architecture of final model to be generated.

    """
    super(MultiChannelNetworkModel, self).__init__()
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
    if primitives is None:
        primitives = PRIMITIVES
    if op_dict is None:
        op_dict = operations.OPS

    self.primitives = primitives
    self.op_dict = op_dict
    self.simple_path = simple_path
    # self.op_types = [operations.SharpSepConv, operations.ResizablePool]
    # Removed condition as it is not required.
    # if self._genotype is not None and type(self._genotype[0]) is np.str_:
    if self.simple_path:
      model = self._genotype[np.flatnonzero(np.core.defchararray.find(self._genotype, 'add') == -1)]
      root_ch = self.Cs[int(model[1][-1])]
      self.stem = nn.ModuleList()
      s = nn.Sequential(
          nn.Conv2d(int(in_channels), root_ch, 3, padding=1, bias=False),
          nn.BatchNorm2d(root_ch))
      self.stem.append(s)
      self.op_grid = nn.ModuleList()
      c_out = 0
      #Switched to primitives and op_dict like Network
      # ops = {'SharpSepConv': 0, 'ResizablePool': 1}

      # Parsing model definition string. Refer genotypes.py for sample model definition string.
      for layers in model[3:-4]:
          layer = layers.split("_")
          # fetching primitive and other parameters from saved model.
          primitive = self.primitives[int(layer[-1])]
          stride = int(layer[3])
          c_in = int(layer[6])
          c_out = int(layer[9])
          op = self.op_dict[primitive](c_in, c_out, stride=stride)
          # Consistent with MixedOp
          if 'pool' in primitive:
              op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
          # Decreasing feature maps so that output is as expected.
          if 'none' in primitive or ('skip_connect' in primitive and stride_idx == 0):
              op = nn.Sequential(op, nn.Conv2d(int(cin), int(cout), 1))
          self.op_grid.append(op)
      self.base = nn.ModuleList()
      self.base.append(operations.SharpSepConv(int(c_out), int(final_linear_filters), 3))
      self.global_pooling = nn.AdaptiveAvgPool2d(1)
      self.classifier = nn.Linear(final_linear_filters, num_classes)

    else:
      self.stem = nn.ModuleList()
      self.stemCs = []

      for i, c in enumerate(self.Cs):
        if "Conv3x3_"+str(i) in self._genotype:
          s = nn.Sequential(
            nn.Conv2d(int(in_channels), int(c), 3, padding=1, bias=False),
            nn.BatchNorm2d(c)
          )
          self.stemCs.append(c)
          self.stem.append(s)

      self.op_grid = nn.ModuleList()
      self.op_grid_list = []

      self.type_modules_list = []
      for layer_idx in range(self._layers):
        stride_modules = nn.ModuleList()
        stride_modules_param = []
        for stride_idx in self.strides:
          out_modules = nn.ModuleList()
          out_modules_param = []
          for C_out_idx in range(self.C_size):
            in_modules = nn.ModuleList()
            in_modules_param = []

            for C_in_idx in range(self.C_size):
              out_node = 'layer_'+str(layer_idx)+'_add_'+'c_out_'+str(self.Cs[C_out_idx])+'_stride_' + str(stride_idx+1)
              type_modules = nn.ModuleList()
              type_modules_list = []

              for primitive_idx, primitive in enumerate(self.primitives):
                  cin = C_in[C_in_idx][C_out_idx]
                  cout = C_out[C_in_idx][C_out_idx]
                  name = 'layer_' + str(layer_idx) + '_stride_' + str(stride_idx+1) + '_c_in_' + str(self.Cs[C_in_idx]) + '_c_out_' + str(self.Cs[C_out_idx]) + '_op_type_' + str(primitive) + '_opid_' + str(primitive_idx)
                  if name in self._genotype:
                    op = self.op_dict[primitive](int(cin), int(cout), int(stride_idx + 1), False)
                    # Consistent with MixedOp
                    if 'pool' in primitive:
                        op = nn.Sequential(op, nn.BatchNorm2d(int(cout), affine=False))
                    # Decreasing feature maps so that output is as expected.
                    if 'none' in primitive or ('skip_connect' in primitive and stride_idx == 0):
                        op = nn.Sequential(op, nn.Conv2d(int(cin), int(cout), 1))
                    type_modules.append(op)
                    type_modules_list.append((primitive_idx, primitive))
                  else:
                    continue
              if len(type_modules) > 0:
                in_modules.append(type_modules)
                in_modules_param.append((self.Cs[C_in_idx], type_modules_list))
            if len(in_modules) > 0:
              out_modules.append(in_modules)

              out_modules_param.append((self.Cs[C_out_idx], in_modules_param))

          # op grid is stride_modules
          if len(out_modules) > 0:
            stride_modules.append(out_modules)
            stride_modules_param.append((stride_idx, out_modules_param))
          # stride_modules_param.append((stride_idx, out_modules_param))
        if len(stride_modules) > 0:
          self.op_grid.append(stride_modules)
          self.op_grid_list.append((layer_idx, stride_modules_param))

      self.base = nn.ModuleList()

      self.baseCs=[]
      for c in self.Cs:
        if "SharpSepConv"+str(c) in self._genotype:

          self.baseCs.append(c)
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


  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input_batch):
    # [in, normal_out, reduce_out]
    if self.simple_path:
      x = input_batch
      for i in range(len(self.stem)):
          x = self.stem[i](x)
      for i in range(len(self.op_grid)):
          x = self.op_grid[i](x)
      out = self.global_pooling(self.base[0](x))
      logits = self.classifier(out.view(out.size(0), -1))
      return logits

    else:

      self.C_size = len(self.stemCs)
      s0s = [[], [None] * self.C_size, [None] * self.C_size]
      for operation in self.stem:
        # Make the set of features with different numbers of channels.
        s0s[0] += [operation(input_batch)]

      # Duplicate s0s to account for 2 different strides
      # s0s += [[]]
      # s1s = [None] * layers + 1

      for layer in self.op_grid_list:
        # layer is how many times we've called everything, i.e. the number of "layers"
        # this is different from the number of layer types which is len([SharpSepConv, ResizablePool]) == 2
        # layer_st_time = time.time()
        layer_idx = layer[0]
        for stride_idx, C_outs in layer[1]:
          stride = 1 + stride_idx

          # C_out_layer = [x[2] for x in self.outCs if x[0] == layer[0] and x[1] == strides[0]]
          # C_outs = strides[1]
          for C_out_grid_id, (C_out, C_ins) in enumerate(C_outs):
            # take all the layers with the same output so we can sum them
            # print('forward layer: ' + str(layer) + ' stride: ' + str(stride) + ' c_out: ' + str(self.Cs[C_out_idx]))
            C_out_idx = np.where(self.Cs == C_out)[0][0]
            c_outs = []
            # C_in_layer = [x[3] for x in self.inCs if x[0] == layer[0] and x[1] == strides[0] and x[2] == C_out]
            for C_in_grid_id, (C_in, primitives) in enumerate(C_ins):
              C_in_idx = np.where(self.Cs == C_in)[0][0]
              for primitive_grid_idx, (primitive_idx, primitive) in enumerate(primitives):

                # get the specific weight for this op
                name = 'layer_' + str(layer_idx) + '_stride_' + str(stride) + '_c_in_' + str(C_in) + '_c_out_' + str(C_out) + '_op_type_' + str(primitive) + '_opid_' + str(primitive_idx)
                
                # layer is present in final model architecture.
                if name in self._genotype:
                  s = s0s[stride_idx][C_in_grid_id]
                  if s is not None:
                    x = self.op_grid[layer_idx][stride_idx][C_in_grid_id][C_out_grid_id][primitive_grid_idx](s)
                    c_outs += [x]

            # only apply updates to layers of sufficient quality
            if c_outs:
              # print('combining c_outs forward layer: ' + str(layer) + ' stride: ' + str(stride) + ' c_out: ' + str(self.Cs[C_out_idx]) + ' c_in: ' + str(self.Cs[C_in_idx]) + ' op type: ' + str(op_type_idx))
              # combined values with the same c_out dimension
              combined = sum(c_outs)
              if s0s[stride][C_out_grid_id] is None:
                # first call sets the value
                s0s[stride][C_out_grid_id] = combined
              else:
                s0s[stride][C_out_grid_id] += combined

        # downscale reduced input as next output
        self.C_out_size = len(C_outs)
        s0s = [s0s[stride], [None] * self.C_out_size, [None] * self.C_out_size]

      # combine results
      # use SharpSepConv to match dimension of final linear layer
      # then add up all remaining outputs and pool the result
      out = self.global_pooling(sum(op(x) for op, x in zip(self.base, s0s[0]) if x is not None))
      logits = self.classifier(out.view(out.size(0), -1))
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

