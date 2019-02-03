# Some data loading code is from https://github.com/DRealArun/darts/ with the same license as darts.
import os
import time
import numpy as np
import logging
import torch
import shutil
import argparse
import glob
import json
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from six import iteritems

from tqdm import tqdm
import colorlog

import autoaugment
import flops_counter


class NumpyEncoder(json.JSONEncoder):
    """ json encoder for numpy types

    source: https://stackoverflow.com/a/49677241/99379
    """
    def default(self, obj):
        if isinstance(obj,
            (np.int_, np.intc, np.intp, np.int8,
             np.int16, np.int32, np.int64, np.uint8,
             np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj,
           (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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


def list_of_dicts_to_dict_of_lists(ld):
    """ list of dictionaries to dictionary of lists when all keys are the same.

    source: https://stackoverflow.com/a/23551944/99379
    """
    return {key: [item[key] for item in ld] for key in ld[0].keys()}


def list_of_dicts_to_csv(filename, list_of_dicts, separator=', ', key_prepend=''):
    headers = []
    values = []
    dict_of_lists = list_of_dicts_to_dict_of_lists(ld)
    for (k, v) in iteritems(dict_of_lists):
        headers += [key_prepend + str(k)]
        values += v

    header = separator.join(headers)
    np.savetxt(filename, values, separator=separator)


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
    """Cutout and dual cutout

    Defaults to Dual Cutout.

    Cutout: https://arxiv.org/abs/1708.04552
    Dual Cutout: https://arxiv.org/pdf/1802.07426

    """
    def __init__(self, length=16, cuts=2):
        self.length = length
        self.cuts = cuts

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.cuts):
          y = np.random.randint(h)
          x = np.random.randint(w)

          y1 = np.clip(y - self.length // 2, 0, h)
          y2 = np.clip(y + self.length // 2, 0, h)
          x1 = np.clip(x - self.length // 2, 0, w)
          x2 = np.clip(x + self.length // 2, 0, w)

          mask[y1: y2, x1: x2] = 0.

        if isinstance(img, torch.Tensor):
          mask = torch.from_numpy(mask)
          mask = mask.expand_as(img)
        img *= mask
        return img



class BatchCutout(object):
  """Cutout and dual cutout

  Defaults to Dual Cutout.

  Cutout: https://arxiv.org/abs/1708.04552
  Dual Cutout: https://arxiv.org/pdf/1802.07426

  """
  def __init__(self, length=16, cuts=2, dtype=np.float32):
      self.length = length
      self.cuts = cuts
      self.dtype = dtype

  def __call__(self, img):
      b, c, h, w = img.shape
      mask = np.ones((b, c, h, w), np.float32)

      for bi in range(b):
        for _ in range(self.cuts):
          y = np.random.randint(h)
          x = np.random.randint(w)

          y1 = np.clip(y - self.length // 2, 0, h)
          y2 = np.clip(y + self.length // 2, 0, h)
          x1 = np.clip(x - self.length // 2, 0, w)
          x2 = np.clip(x + self.length // 2, 0, w)

          mask[bi, :, y1: y2, x1: x2] = 0.

      if isinstance(img, torch.Tensor):
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
      img *= mask
      return img


# Function to fetch the transforms based on the dataset
def get_data_transforms(args, normalize_as_tensor=True):
  """Get the transforms for a specific dataset

  One side side effect args.std and args.mean are set.

  args: parser args. Expected to have random_eraser, cutout,
    and autoaugment member variables.
  normalize_as_tensor: when true the output will be converted
    to a tensor then normalization will be applied based on the
    dataset mean and std dev. Otherwise this step will be skipped
    entirely

  """
  print("get_data_transforms(): Getting ", args.dataset, " Transforms")
  if args.dataset == 'cifar10':
    return _data_transforms_cifar10(args, normalize_as_tensor)
  if args.dataset == 'mnist':
    return _data_transforms_mnist(args, normalize_as_tensor)
  if args.dataset == 'emnist':
    return _data_transforms_emnist(args, normalize_as_tensor)
  if args.dataset == 'fashion':
    return _data_transforms_fashion(args, normalize_as_tensor)
  if args.dataset == 'svhn':
    return _data_transforms_svhn(args, normalize_as_tensor)
  if args.dataset == 'stl10':
    return _data_transforms_stl10(args, normalize_as_tensor)
  if args.dataset == 'devanagari':
    return _data_transforms_devanagari(args, normalize_as_tensor)
  if args.dataset == 'imagenet':
    return _data_transforms_imagenet(args, normalize_as_tensor)
  assert False, "Cannot get Transform for dataset"


def finalize_transform(train_transform, valid_transform, args, normalize_as_tensor=True):
  """ Transform steps that apply to most augmentation regimes
  """
  if normalize_as_tensor:
    # train
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(
      transforms.Normalize(args.mean, args.std))
    # valid
    valid_transform.transforms.append(transforms.ToTensor())
    valid_transform.transforms.append(
      transforms.Normalize(args.mean, args.std))
    # note that the current cutout and random eraser implementations
    # require tensors as imput, so don't get applied when
    # normalize_as_tensor is False

    # cutout should be after normalize
    if args.cutout:
      # note that this defaults to dual cutout
      train_transform.transforms.append(Cutout(args.cutout_length))
    if args.random_eraser:
      train_transform.transforms.append(random_eraser)
  return train_transform, valid_transform


# Transform defined for imagenet
def _data_transforms_imagenet(args, normalize_as_tensor=True):
  IMAGENET_MEAN = [0.485, 0.456, 0.406]
  IMAGENET_STD = [0.229, 0.224, 0.225]
  args.mean = IMAGENET_MEAN
  args.std = IMAGENET_MEAN

  if(args.arch == "inception_v3"):
    crop_size = 299
    val_size = 320  # nvidia author chose this value arbitrarily, we can adjust.
  else:
    crop_size = 224
    val_size = 256
  if args.autoaugment:
    train_transform = transforms.Compose([
      transforms.RandomResizedCrop(crop_size),
      transforms.RandomHorizontalFlip(),
      # cutout and autoaugment are used in the autoaugment paper
      autoaugment.ImageNetPolicy(),
    ])
  else:
    train_transform = transforms.Compose([
      transforms.RandomResizedCrop(crop_size),
      transforms.RandomHorizontalFlip(),
    ])

  valid_transform = transforms.Compose([
    transforms.Resize(val_size),
    transforms.CenterCrop(crop_size)
  ])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for cifar-10
def _data_transforms_cifar10(args, normalize_as_tensor=True):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
  args.mean = CIFAR_MEAN
  args.std = CIFAR_STD
  if args.autoaugment:
    train_transform = transforms.Compose([
      # NOTE(ahundt) pad and fill has been added to support autoaugment. Results may have changed! https://github.com/DeepVoltaire/AutoAugment/issues/8
      transforms.Pad(4, fill=128),
      transforms.RandomCrop(32, padding=0),
      transforms.RandomHorizontalFlip(),
      autoaugment.CIFAR10Policy(),
    ])
  else:
    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
    ])

  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for mnist
def _data_transforms_mnist(args, normalize_as_tensor=True):
  MNIST_MEAN = (0.1307,)
  MNIST_STD = (0.3081,)
  args.mean = MNIST_MEAN
  args.std = MNIST_STD

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
  ])
  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for fashion mnist
def _data_transforms_fashion(args, normalize_as_tensor=True):
  FASHION_MEAN = (0.2860405969887955,)
  FASHION_STD = (0.35302424825650003,)
  args.mean = FASHION_MEAN
  args.std = FASHION_STD

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(FASHION_MEAN, FASHION_STD),
  ])
  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for emnist
def _data_transforms_emnist(args, normalize_as_tensor=True):
  EMNIST_MEAN = (0.17510417052459282,)
  EMNIST_STD = (0.33323714976320795,)
  args.mean = EMNIST_MEAN
  args.std = EMNIST_STD

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
  ])
  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for svhn
def _data_transforms_svhn(args, normalize_as_tensor=True):
  SVHN_MEAN = [ 0.4376821,   0.4437697,   0.47280442]
  SVHN_STD = [ 0.19803012,  0.20101562,  0.19703614]
  args.mean = SVHN_MEAN
  args.std = SVHN_STD

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for stl10
def _data_transforms_stl10(args, normalize_as_tensor=True):
  STL_MEAN = [ 0.44671062,  0.43980984,  0.40664645]
  STL_STD = [ 0.26034098,  0.25657727,  0.27126738]
  args.mean = STL_MEAN
  args.std = STL_STD

  train_transform = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(STL_MEAN, STL_STD),
  ])
  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


# Transform defined for devanagari hand written symbols
def _data_transforms_devanagari(args, normalize_as_tensor=True):
  DEVANAGARI_MEAN = (0.240004663268,)
  DEVANAGARI_STD = (0.386530114768,)
  args.mean = DEVANAGARI_MEAN
  args.std = DEVANAGARI_STD

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=2), #Already has padding 2 and size is 32x32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
  ])
  valid_transform = transforms.Compose([])
  return finalize_transform(train_transform, valid_transform, args, normalize_as_tensor)


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def count_model_flops(cnn_model, data_shape=[1, 3, 32, 32]):
  cnn_model_flops = cnn_model
  batch = torch.zeros(data_shape)
  if torch.cuda.is_available():
    batch = batch.cuda()
  cnn_model_flops = flops_counter.add_flops_counting_methods(cnn_model)
  cnn_model_flops.eval().start_flops_count()
  out = cnn_model_flops(batch)
  cnn_model_flops.stop_flops_count()
  flops_str = flops_counter.flops_to_string(cnn_model.compute_average_flops_cost())
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


def initialize_files_and_args(args, run_type='eval'):
  """Adds parameters to args and creates the folder for the log and weights with a code backup as needed.

  This function is pretty data loader and argument specific,
  and thus a bit brittle and not intended for general use.
  Loads args from a file if specified by the user, args may change substantially!
  This happens particularly when args.load_args or args.evaluate is set.
  Creates the log folder if it does not exist.

  Input:
  args.evaluate: empty string or path to a weights file to evaluate
  args.load_args: json file containing saved command line arguments which will be loaded.
  args.save: custom name to give the log folder so you know what this run is about.
  args.gpu: the integer id of the gpu on which to run.
  args.dataset: a string with the name of the dataset.
  args.arch: a string with the name of the neural network architecture being used.

  Output:
  args.stats_file: full path to file for final json statistics
  args.epoch_stats_file: full path to file for json with per-epoch statistics
  args.save: new save directory, or existing directory if evaluating.
  args.evaluate: are we doing an evaluation-only run
  args.load: updated if a weights file was specified via args.evaluate
  args.log_file_path: set with the path to the file where logs will be written.
     This variable is designed to be passed to utils.logging_setup(log_file_path).

  Returns:

  updated args object
  """
  log_file_name = 'log.txt'

  evaluate_arg = args.evaluate
  loaded_args = False
  if args.load_args:
    with open(args.load_args, 'r') as f:
      args_dict = vars(args)
      args_dict.update(json.load(f))
      args = argparse.Namespace(**args_dict)
    args.evaluate = evaluate_arg
    loaded_args = True

  stats_time = time.strftime("%Y%m%d-%H%M%S")
  if evaluate_arg:
    # evaluate results go in the same directory as the weights but with a new timestamp
    # we will put the logs in the same directory as the weights
    save_dir = os.path.dirname(os.path.realpath(evaluate_arg))
    log_file_name = 'eval-log-' + stats_time + '.txt'
    log_file_path = os.path.join(save_dir, log_file_name)
    params_path = os.path.join(save_dir, 'commandline_args.json')
    if not loaded_args:
      print('Warning: --evaluate specified, loading commandline args from:\n' + params_path)
      with open(params_path, 'r') as f:
        args_dict = vars(args)
        args_dict.update(json.load(f))
        args = argparse.Namespace(**args_dict)
    args.evaluate = evaluate_arg
    args.load = evaluate_arg
    args.save = save_dir

  else:
    args.save = '{}-{}-{}-{}-{}-{}'.format(run_type, stats_time, args.save, args.dataset, args.arch, args.gpu)
    params_path = os.path.join(args.save, 'commandline_args.json')
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_file_path = os.path.join(args.save, log_file_name)
    with open(params_path, 'w') as f:
        json.dump(vars(args), f)

  stats_file_name = 'eval-stats-' + stats_time + '.json'
  args.epoch_stats_file = os.path.join(args.save, 'eval-epoch-stats-' + stats_time + '.json')
  args.stats_file = os.path.join(args.save, stats_file_name)
  args.log_file_path = log_file_path
  return args