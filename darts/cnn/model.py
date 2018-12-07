import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from . import operations
from . import genotypes
from .operations import ReLUConvBN
from .operations import ConvBNReLU
from .operations import FactorizedReduce
from torch.autograd import Variable
from .utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, op_dict=None):
    """Create a final cell with a single architecture.

    The Cell class in model_search.py is the equivalent for searching multiple architectures.

    # Arguments

      op_dict: The dictionary of possible operation creation functions.
        All primitive name strings defined in the genotype must be in the op_dict.
    """
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    if op_dict is None:
          op_dict = operations.OPS
    # _op_dict are op_dict available for use,
    # _ops is the actual sequence of op_dict being utilized in this case
    self._op_dict = op_dict

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = self._op_dict[name](C, stride, True)
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
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
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

  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=3, reduce_spacing=None):
    """
    # Arguments

        C: Initial number of output channels.
        in_channels: initial number of input channels
        layers: The number of cells to create.
        reduce_spacing: number of layers of cells between reduction cells,
            default of None is at 1/3 and 2/3 of the total number of layers.
            1 means all cells are reduction. 2 means the first layer is
            normal then the second
    """
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels

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
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=3):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels

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
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux




# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  """
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
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

    return C_prev_prev, C_prev

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self._drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    if self._auxiliary:
      return logits, logits_aux
    else:
      #TODO(ahundt) previously a tuple was always returned and this if statement wasn't here before. check for compatibility with other DARTS code
      return logits

  def reset_noise(self):
        self.classifier.reset_noise()
