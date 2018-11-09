import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
try:
    import fanova
except ImportError:
    print('fanova not available, skipping components ranking hyperparam importance, try:'
          '    pip install fanova --user --upgrade'
          'or follow the instructions at https://github.com/automl/fanova')
    fanova = None

from tqdm import tqdm

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def logging_setup(log_file_path):
    # setup logging for tqdm compatibility
    # based on https://github.com/tqdm/tqdm/issues/193#issuecomment-232887740
    logger = colorlog.getLogger("SQUARE")
    logger.setLevel(logging.DEBUG)
    handler = TqdmHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y_%m_%d_%H_%M_%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'SUCCESS:': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'},))

    logger.addHandler(handler)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addhandler(fh)
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

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
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
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


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


class Performance(object):
  def __init__(self, path):
    self.path = path
    self.data = None

  def update(self, alphas_normal, alphas_reduce, val_loss):
    a_normal = F.softmax(alphas_normal, dim=-1)
    # print("alpha normal size: ", a_normal.data.size())
    a_reduce = F.softmax(alphas_reduce, dim=-1)
    # print("alpha reduce size: ", a_reduce.data.size())
    data = np.concatenate([a_normal.data.view(-1),
                           a_reduce.data.view(-1),
                           np.array([val_loss.data])]).reshape(1,-1)
    if self.data is not None:
      self.data = np.concatenate([self.data, data], axis=0)
    else:
      self.data = data

  def save(self):
    np.save(self.path, self.data)

def importance(path, config):
  assert os.path.exists(path), 'File %s does not exist' %path
  assert isinstance(config, dict), 'Input argument config is wrong'

  data = np.load(path)
  X = data[:, :-1].astype(np.double)
  Y = data[:, -1].astype(np.double)
  n_data, n_params = X.shape
  print(X.shape)
  imps = []
  # TODO: make dims, max_cols more clear & automatically deduce them, may be related to search space
  dims = (10, )
  max_cols = 50

  if config['mode'] == 'incremental':
    interval = config['interval']
    for i in range(n_data // interval):
      print('Iteration %d: \n' %i)
      f = fanova.fANOVA(X[:(i+1)*interval, :max_cols], Y[:(i+1)*interval])
      imp_dic = f.quantify_importance(dims)
      print(imp_dic)
      imps.append(imp_dic)
  elif config['mode'] == 'fixed':
    interval = config['interval']
    for i in range(n_data // interval):
      print('Iteration %d: \n' %i)
      f = fanova.fANOVA(X[i*interval:(i+1)*interval, :max_cols], Y[i*interval:(i+1)*interval])
      imp_dic = f.quantify_importance(dims)
      print(imp_dic)
      imps.append(imp_dic)
  return imps

# if __name__ == '__main__':
#   path = '/home/zero/Downloads/cifar10_performance.npy'
#   config = {'mode': 'fixed', 'interval': 1000}
#   # config = {'mode': 'incremental', 'interval': 1000}
#   imps = importance(path, config)
