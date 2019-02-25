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
from model import Cell
from model import AuxiliaryHeadImageNet
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class NetworkResNetCOSTAR(nn.Module):
  # Baseline model based on https://arxiv.org/pdf/1611.08036.pdf
  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=6, reduce_spacing=None,
               mixed_aux=False, op_dict=None, C_mid=None, stem_multiplier=3, vector_size=15):
    super(NetworkResNetCOSTAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels
    self._vector_size = vector_size
    self.drop_path_prob = 0.
    resnet_linear_count = 2048
    pretrained = True
    self.stem0 = models.resnet50(num_classes=resnet_linear_count)
    # self.stem1 = models.resnet50(num_classes=resnet_linear_count)
    if pretrained:
        weights = model_zoo.load_url(models.resnet.model_urls['resnet50'])
        # remove weights which we will not be loading
        del weights['fc.weight']
        del weights['fc.bias']
        # load pretrained weights
        self.stem0.load_state_dict(weights, strict=False)
        # self.stem1.load_state_dict(weights, strict=False)
    self.global_pooling = nn.AvgPool2d(7)

    # if auxiliary:
    #   self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    # self.global_pooling = nn.AvgPool2d(7)
    # Input minus the image channels
    combined_state_size = resnet_linear_count + resnet_linear_count + vector_size
    # print('>>>>> C:' + str(C) + ' combined state size: ' + str(combined_state_size))
    # print('>>>>> C:' + str(C) + ' combined state size: ' + str(combined_state_size))
    
    self.classifier = nn.Sequential(
        nn.Linear(combined_state_size, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

  def forward(self, img, vec):
    logits_aux = None
    # print('vector_in: ' + str(vector_in.data))
    # print('pixel_sample: ' + str(batch_input[:,:,1,1]))
    s0 = self.stem0(img[:, :3, :, :])
    s1 = self.stem0(img[:, 3:, :, :])
    # x = torch.cat([s0, s1], dim=1)
    # x = self.global_pooling(x)
    # vector_in = batch_input[:,6:,1,1]
    # out = torch.cat([x, vector_in], dim=-1)
    out = torch.cat([s0, s1, vec], dim=-1)
    # print('>>>>>>> out shape: ' + str(out.shape))
    # print('>>>>>>> out shape: ' + str(out.shape))
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux


class NetworkNASCOSTAR(nn.Module):
  # Copied from NetworkImageNet
  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=55, reduce_spacing=None,
               mixed_aux=False, op_dict=None, C_mid=None, stem_multiplier=3):
    super(NetworkNASCOSTAR, self).__init__()
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
    self.classifier = nn.Linear(C_prev, C_prev)
    self.classifier2 = nn.Linear(C_prev, num_classes)

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


class NetworkCOSTAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, in_channels=6, reduce_spacing=None,
               mixed_aux=False, op_dict=None, C_mid=None, stem_multiplier=3, vector_size=15):
    super(NetworkCOSTAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._in_channels = in_channels
    self._vector_size = vector_size
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

    self.vector_stem = nn.Sequential(
      nn.Linear(vector_size, C),
      nn.ReLU(inplace=True),
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

  def forward(self, img, vector):
    logits_aux = None
    s0 = self.stem0(img)
    s1 = self.stem1(s0)
    v = self.vector_stem(vector)

    s1.add_(v.unsqueeze(2).unsqueeze(3).expand(-1, -1, s1.shape[2], s1.shape[3]))

    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux