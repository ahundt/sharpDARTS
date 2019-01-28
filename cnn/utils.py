# Some data loading code is from https://github.com/DRealArun/darts/ with the same license as darts.
import os
import numpy as np
import logging
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from six import iteritems

from tqdm import tqdm
import colorlog

import autoaugment
import flops_counter


def tqdm_stats(progbar, prefix=''):
  """ Very brittle function to extract timing stats from tqdm.
  Replace when https://github.com/tqdm/tqdm/issues/562 is resolved.
  Example of key string component that will be read:
     3/3 [00:00<00:00, 12446.01it/s]
  """
  s = str(progbar)
  # get the stats part of the string
  s = s[s.find("| ")+1:]
  stats = {
    prefix + 'current_step': s[:s.find('/')].strip(' '),
    prefix + 'total_steps': s[s.find('/')+1:s.find('[')].strip(' '),
    prefix + 'time_elapsed': s[s.find('[')+1:s.find('<')].strip(' '),
    prefix + 'time_remaining': s[s.find('<')+1:s.find(',')].strip(' '),
    prefix + 'step_time': s[s.find(', ')+1:s.find(']')].strip(' '),
  }
  if '%' in s:
    stats[prefix + 'percent_complete'] = s[:s.find('%')].strip(' ')
  return stats

def dict_to_log_string(log={}, separator=', ', key_prepend=''):
  log_strings = []
  for (k, v) in iteritems(log):
    log_strings += [key_prepend + str(k), str(v)]
  return separator.join(log_strings)


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def logging_setup(log_file_path):
    """ setup logging to a file and support for tqdm progress bar

      log_file_path: path to log file which will be created as a txt to output printed information
    """
    # setup logging for tqdm compatibility
    # based on https://github.com/tqdm/tqdm/issues/193#issuecomment-232887740
    logger = colorlog.getLogger("SQUARE")
    logger.setLevel(logging.DEBUG)
    handler = TqdmHandler()
    log_format = colorlog.ColoredFormatter(
        # '%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s',
        '%(asctime)s %(message)s',
        datefmt='%Y_%m_%d_%H_%M_%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'SUCCESS:': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'},)
    handler.setFormatter(log_format)

    logger.addHandler(handler)
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    return logger


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def random_eraser(input_img, p=0.66, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=True):
    """ Cutout and random erasing algorithms for data augmentation

    source:
    https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py

    modified for batch, channel, height, width dimension order, and so there are no while loop delays.
    """
    img_c, img_h, img_w = input_img.shape
    # print('input_img.shape' + str(input_img.shape))
    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    s = np.random.uniform(s_l, s_h) * img_h * img_w
    r = np.random.uniform(r_1, r_2)
    w = int(np.sqrt(s / r))
    h = int(np.sqrt(s * r))
    left = np.random.randint(0, img_w)
    top = np.random.randint(0, img_h)
    # ensure boundaries fit in the image border
    w = np.clip(w, 0, img_w - left)
    h = np.clip(h, 0, img_h - top)

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (img_c, h, w))
    else:
        c = np.random.uniform(v_l, v_h)

    c = torch.from_numpy(c)

    # print('c.shape' + str(c.shape))
    input_img[:, top:top + h, left:left + w] = c

    return input_img


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


# Function to fetch the transforms based on the dataset
def get_data_transforms(args):
  print("Getting",args.dataset,"Transforms")
  if args.dataset == 'cifar10':
    return _data_transforms_cifar10(args)
  if args.dataset == 'mnist':
    return _data_transforms_mnist(args)
  if args.dataset == 'emnist':
    return _data_transforms_emnist(args)
  if args.dataset == 'fashion':
    return _data_transforms_fashion(args)
  if args.dataset == 'svhn':
    return _data_transforms_svhn(args)
  if args.dataset == 'stl10':
    return _data_transforms_stl10(args)
  if args.dataset == 'devanagari':
    return _data_transforms_devanagari(args)
  assert False, "Cannot get Transform for dataset"

# Transform defined for cifar-10
def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
  if args.autoaugment:
    train_transform = transforms.Compose([
      # NOTE(ahundt) pad and fill has been added to support autoaugment. Results may have changed! https://github.com/DeepVoltaire/AutoAugment/issues/8
      transforms.Pad(4, fill=128),
      transforms.RandomCrop(32, padding=0),
      transforms.RandomHorizontalFlip(),
      autoaugment.CIFAR10Policy(),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
  else:
    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


# Transform defined for mnist
def _data_transforms_mnist(args):
  MNIST_MEAN = (0.1307,)
  MNIST_STD = (0.3081,)

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
  ])
  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
  return train_transform, valid_transform


# Transform defined for fashion mnist
def _data_transforms_fashion(args):
  FASHION_MEAN = (0.2860405969887955,)
  FASHION_STD = (0.35302424825650003,)

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(FASHION_MEAN, FASHION_STD),
  ])
  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(FASHION_MEAN, FASHION_STD),
    ])
  return train_transform, valid_transform


# Transform defined for emnist
def _data_transforms_emnist(args):
  EMNIST_MEAN = (0.17510417052459282,)
  EMNIST_STD = (0.33323714976320795,)

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
  ])
  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])
  return train_transform, valid_transform


# Transform defined for svhn
def _data_transforms_svhn(args):
  SVHN_MEAN = [ 0.4376821,   0.4437697,   0.47280442]
  SVHN_STD = [ 0.19803012,  0.20101562,  0.19703614]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
  return train_transform, valid_transform


# Transform defined for stl10
def _data_transforms_stl10(args):
  STL_MEAN = [ 0.44671062,  0.43980984,  0.40664645]
  STL_STD = [ 0.26034098,  0.25657727,  0.27126738]

  train_transform = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(STL_MEAN, STL_STD),
  ])
  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(STL_MEAN, STL_STD),
    ])
  return train_transform, valid_transform


# Transform defined for devanagari hand written symbols
def _data_transforms_devanagari(args):
  DEVANAGARI_MEAN = (0.240004663268,)
  DEVANAGARI_STD = (0.386530114768,)

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=2), #Already has padding 2 and size is 32x32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
  ])
  if args.random_eraser:
    train_transform.transforms.append(random_eraser)
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def count_model_flops(cnn_model, data_shape=[1, 3, 32, 32]):
  cnn_model_flops = cnn_model.clone()
  cnn_model_flops = flops_counter.add_flops_counting_methods(cnn_model)
  batch = torch.zeros(data_shape)
  cnn_model_flops.eval.start_flops_count()
  out = cnn_model_flops(batch)
  cnn_model_flops.stop_flops_count()
  flops_str = flops_to_string(model.compute_average_flops_cost())
  del cnn_model_flops
  del batch
  return flops_str


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
