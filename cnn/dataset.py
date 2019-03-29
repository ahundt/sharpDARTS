import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
try:
  import costar_dataset
except ImportError:
  print('dataset.py: The costar dataset is not available, so it is being skipped. '
        'See https://github.com/ahundt/costar_dataset for details')
  costar_dataset = None

CIFAR_CLASSES = 10
MNIST_CLASSES = 10
FASHION_CLASSES = 10
EMNIST_CLASSES = 47
SVHN_CLASSES = 10
STL10_CLASSES = 10
DEVANAGARI_CLASSES = 46
IMAGENET_CLASSES = 1000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
FASHION_MEAN = (0.2860405969887955,)
FASHION_STD = (0.35302424825650003,)
EMNIST_MEAN = (0.17510417052459282,)
EMNIST_STD = (0.33323714976320795,)
SVHN_MEAN = [ 0.4376821,   0.4437697,   0.47280442]
SVHN_STD = [ 0.19803012,  0.20101562,  0.19703614]
STL10_MEAN = [ 0.44671062,  0.43980984,  0.40664645]
STL10_STD = [ 0.26034098,  0.25657727,  0.27126738]
DEVANAGARI_MEAN = (0.240004663268,)
DEVANAGARI_STD = (0.386530114768,)

class_dict = {'cifar10': CIFAR_CLASSES,
              'mnist' : MNIST_CLASSES,
              'emnist': EMNIST_CLASSES,
              'fashion': FASHION_CLASSES,
              'svhn': SVHN_CLASSES,
              'stl10': STL10_CLASSES,
              'devanagari' : DEVANAGARI_CLASSES,
              'imagenet' : IMAGENET_CLASSES}

mean_dict = {'cifar10': CIFAR_MEAN,
             'mnist' : MNIST_MEAN,
             'emnist': EMNIST_MEAN,
             'fashion': FASHION_MEAN,
             'svhn': SVHN_MEAN,
             'stl10': STL10_MEAN,
             'devanagari' : DEVANAGARI_MEAN,
             'imagenet' : IMAGENET_MEAN}

std_dict = {'cifar10': CIFAR_STD,
            'mnist' : MNIST_STD,
            'emnist': EMNIST_STD,
            'fashion': FASHION_STD,
            'svhn': SVHN_STD,
            'stl10': STL10_STD,
            'devanagari' : DEVANAGARI_STD,
            'imagenet' : IMAGENET_STD}

inp_channel_dict = {'cifar10': 3,
                    'mnist' : 1,
                    'emnist': 1,
                    'fashion': 1,
                    'svhn': 3,
                    'stl10': 3,
                    'devanagari' : 1,
                    'imagenet': 3}

costar_class_dict = {'translation_only': 3,
                     'rotation_only': 5,
                     'all_features': 8}
costar_supercube_inp_channel_dict = {'translation_only': 52,
                           'rotation_only': 55,
                           'all_features': 57}
costar_vec_size_dict = {'translation_only': 44,
                        'rotation_only': 49,
                        'all_features': 49}

COSTAR_SET_NAMES = ['blocks_only', 'blocks_with_plush_toy']
COSTAR_SUBSET_NAMES = ['success_only', 'error_failure_only', 'task_failure_only', 'task_and_error_failure']

def get_training_queues(dataset_name, train_transform, valid_transform, dataset_location=None, batch_size=32, train_proportion=1.0, search_architecture=False,
                        costar_version='v0.4', costar_set_name=None, costar_subset_name=None, costar_feature_mode=None, costar_output_shape=(224, 224, 3),
                        costar_random_augmentation=None, costar_one_hot_encoding=True, costar_num_images_per_example=200, distributed=False, num_workers=12,
                        collate_fn=torch.utils.data.dataloader.default_collate, verbose=0, evaluate=False):
  print("Getting " + dataset_name + " data")
  if dataset_name == 'imagenet':
    print("Using IMAGENET training set")
    # first check if we are just one directory above the imagenet dir
    # imagenet_dir = os.path.join(dataset_location, 'imagenet')
    # if os.path.exists(imagenet_dir):
    #   dataset_location = imagenet_dir
    # set the train directory
    train_dir = os.path.join(dataset_location, 'train')
    train_data = dset.ImageFolder(train_dir, train_transform)
  elif dataset_name == 'cifar10':
    print("Using CIFAR10 training set")
    train_data = dset.CIFAR10(root=dataset_location, train=True, download=True, transform=train_transform)
  elif dataset_name == 'mnist':
    print("Using MNIST training set")
    train_data = dset.MNIST(root=dataset_location, train=True, download=True, transform=train_transform)
  elif dataset_name == 'emnist':
    print("Using EMNIST training set")
    train_data = dset.EMNIST(root=dataset_location, split='balanced', train=True, download=True, transform=train_transform)
  elif dataset_name == 'fashion':
    print("Using Fashion training set")
    train_data = dset.FashionMNIST(root=dataset_location, train=True, download=True, transform=train_transform)
  elif dataset_name == 'svhn':
    print("Using SVHN training set")
    train_data = dset.SVHN(root=dataset_location, split='train', download=True, transform=train_transform)
  elif dataset_name == 'stl10':
    print("Using STL10 training set")
    train_data = dset.STL10(root=dataset_location, split='train', download=True, transform=train_transform)
  elif dataset_name == 'devanagari':
    print("Using DEVANAGARI training set")
    def grey_pil_loader(path):
      # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
      with open(path, 'rb') as f:
          img = Image.open(f)
          img = img.convert('L')
          return img
    # Ensure dataset is present in the directory args.data. Does not support auto download
    train_data = dset.ImageFolder(root=dataset_location, transform=train_transform, loader = grey_pil_loader)
  elif dataset_name == 'stacking':
    # Support for costar block stacking generator implemented by Chia-Hung Lin (rexxarchl)
    # sites.google.com/costardataset
    # https://github.com/ahundt/costar_dataset
    # https://sites.google.com/site/costardataset
    if costar_dataset is None:
      raise ImportError("Trying to use costar_dataset but it was not imported")

    print("Using CoSTAR Dataset")
    if costar_set_name is None or costar_set_name not in COSTAR_SET_NAMES:
      raise ValueError("Specify costar_set_name as one of {'blocks_only', 'blocks_with_plush_toy'}")
    if costar_subset_name is None or costar_subset_name not in COSTAR_SUBSET_NAMES:
      raise ValueError("Specify costar_subset_name as one of {'success_only', 'error_failure_only', 'task_failure_only', 'task_and_error_failure'}")

    train_data = costar_dataset.CostarBlockStackingDataset.from_standard_txt(
                      root=dataset_location, single_batch_cube=False,
                      version=costar_version, set_name=costar_set_name, subset_name=costar_subset_name,
                      split='train', feature_mode=costar_feature_mode, output_shape=costar_output_shape,
                      random_augmentation=costar_random_augmentation, one_hot_encoding=costar_one_hot_encoding,
                      verbose=verbose, num_images_per_example=costar_num_images_per_example, is_training=not evaluate)

  else:
    assert False, "Cannot get training queue for dataset"

  num_train = len(train_data)
  indices = list(range(num_train))
  if search_architecture:
    # select the 'validation' set from the training data
    split = int(np.floor(train_proportion * num_train))
    print("search_architecture enabled, splitting training set into train and val.")
    print("Total Training size", num_train)
    print("Training set size", split)
    print("Training subset for validation size", num_train-split)
    valid_data = train_data
  else:
    split = num_train
    # get the actual train/test set
    if dataset_name == 'imagenet':
        print("Using IMAGENET validation data")
        valid_dir = os.path.join(dataset_location, 'val')
        valid_data = dset.ImageFolder(valid_dir, valid_transform)
    elif dataset_name == 'cifar10':
        print("Using CIFAR10 validation data")
        valid_data = dset.CIFAR10(root=dataset_location, train=search_architecture, download=True, transform=valid_transform)
    elif dataset_name == 'mnist':
        print("Using MNIST validation data")
        valid_data = dset.MNIST(root=dataset_location, train=search_architecture, download=True, transform=valid_transform)
    elif dataset_name == 'emnist':
        print("Using EMNIST validation data")
        valid_data = dset.EMNIST(root=dataset_location, split='balanced', train=search_architecture, download=True, transform=valid_transform)
    elif dataset_name == 'fashion':
        print("Using Fashion validation data")
        valid_data = dset.FashionMNIST(root=dataset_location, train=search_architecture, download=True, transform=valid_transform)
    elif dataset_name == 'svhn':
        print("Using SVHN validation data")
        valid_data = dset.SVHN(root=dataset_location, split='test', download=True, transform=valid_transform)
    elif dataset_name == 'stl10':
        print("Using STL10 validation data")
        valid_data = dset.STL10(root=dataset_location, split='test', download=True, transform=valid_transform)
    elif dataset_name == 'devanagari':
        print("Using DEVANAGARI validation data")
        def grey_pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
          with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            return img
        # Ensure dataset is present in the directory args.data. Does not support auto download
        valid_data = dset.ImageFolder(root=dataset_location, transform=valid_transform, loader = grey_pil_loader)
    elif dataset_name == 'stacking':
        valid_data = costar_dataset.CostarBlockStackingDataset.from_standard_txt(
                          root=dataset_location, single_batch_cube=False,
                          version=costar_version, set_name=costar_set_name, subset_name=costar_subset_name,
                          split='val', feature_mode=costar_feature_mode, output_shape=costar_output_shape,
                          random_augmentation=costar_random_augmentation, one_hot_encoding=costar_one_hot_encoding,
                          verbose=verbose, num_images_per_example=costar_num_images_per_example, is_training=False)
    else:
        assert False, "Cannot get training queue for dataset"

  if dataset_name == 'devanagari':
    print("SHUFFLE INDEX LIST BEFORE BATCHING")
    print("Before Shuffle", indices[-10:num_train])
    np.random.shuffle(indices)
    print("After Shuffle", indices[-10:num_train])

  if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(indices[:split])
  elif evaluate:
    print("Evaluate mode! Training set will appear sequentially.")
    train_sampler = None  # Use default sampler, i.e. Sequential Sampler, when in evaluation
  else:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
  # shuffle does not need to be set to True because
  # that is taken care of by the subset random sampler
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=batch_size,
      sampler=train_sampler,
      pin_memory=True, num_workers=num_workers,
      collate_fn=collate_fn)

  if search_architecture:
    # validation sampled from training set
    val_from_train_indices = indices[split:num_train]
    if distributed:
      valid_sampler = torch.utils.data.distributed.DistributedSampler(val_from_train_indices)
    else:
      valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_from_train_indices)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=num_workers,
        collate_fn=collate_fn)
  else:
    # test set
    valid_sampler = None
    if distributed:
      valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=num_workers,
        collate_fn=collate_fn)

  return train_queue, valid_queue


def get_costar_test_queue(dataset_location, costar_set_name, costar_subset_name, costar_version='v0.4', costar_feature_mode=None, costar_output_shape=(224, 224, 3),
                          costar_random_augmentation=None, costar_one_hot_encoding=True, costar_num_images_per_example=200, batch_size=32, verbose=0,
                          collate_fn=torch.utils.data.dataloader.default_collate):
  # Support for costar block stacking generator implemented by Chia-Hung Lin (rexxarchl)
  # sites.google.com/costardataset
  # https://github.com/ahundt/costar_dataset
  # https://sites.google.com/site/costardataset
  if costar_dataset is None:
    raise ImportError("Trying to use costar_dataset but it was not imported")

  if verbose > 0:
    print("Getting CoSTAR BSD test set...")

  if costar_set_name not in COSTAR_SET_NAMES:
    raise ValueError("Specify costar_set_name as one of {'blocks_only', 'blocks_with_plush_toy'}")
  if costar_subset_name not in COSTAR_SUBSET_NAMES:
    raise ValueError("Specify costar_subset_name as one of {'success_only', 'error_failure_only', 'task_failure_only', 'task_and_error_failure'}")

  test_data = costar_dataset.CostarBlockStackingDataset.from_standard_txt(
                  root=dataset_location, single_batch_cube=False,
                  version=costar_version, set_name=costar_set_name, subset_name=costar_subset_name,
                  split='test', feature_mode=costar_feature_mode, output_shape=costar_output_shape,
                  random_augmentation=costar_random_augmentation, one_hot_encoding=costar_one_hot_encoding,
                  verbose=verbose, num_images_per_example=costar_num_images_per_example, is_training=False)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=batch_size, collate_fn=collate_fn,
      pin_memory=False, num_workers=4)

  return test_queue
