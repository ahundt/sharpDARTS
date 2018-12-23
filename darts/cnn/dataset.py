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
  print('dataset.py: The costar dataset is not available, so it is being skipped.'
        'see https://github.com/ahundt/costar_dataset for details')
  costar_dataset = None

CIFAR_CLASSES = 10
MNIST_CLASSES = 10
FASHION_CLASSES = 10
EMNIST_CLASSES = 47
SVHN_CLASSES = 10
STL10_CLASSES = 10
DEVANAGARI_CLASSES = 46

class_dict = {'cifar10': CIFAR_CLASSES,
              'mnist' : MNIST_CLASSES,
              'emnist': EMNIST_CLASSES,
              'fashion': FASHION_CLASSES,
              'svhn': SVHN_CLASSES,
              'stl10': STL10_CLASSES,
              'devanagari' : DEVANAGARI_CLASSES}

inp_channel_dict = {'cifar10': 3,
                    'mnist' : 1,
                    'emnist': 1,
                    'fashion': 1,
                    'svhn': 3,
                    'stl10': 3,
                    'devanagari' : 1}

def get_training_queues(dataset_name, train_transform, dataset_location=None, batch_size=32, train_proportion=0.9, search_architecture=True):
  print("Getting " + dataset_name + " data")
  if dataset_name == 'cifar10':
    print("Using CIFAR10")
    train_data = dset.CIFAR10(root=dataset_location, train=True, download=True, transform=train_transform)
  elif dataset_name == 'mnist':
    print("Using MNIST")
    train_data = dset.MNIST(root=dataset_location, train=True, download=True, transform=train_transform)
  elif dataset_name == 'emnist':
    print("Using EMNIST")
    train_data = dset.EMNIST(root=dataset_location, split='balanced', train=True, download=True, transform=train_transform)
  elif dataset_name == 'fashion':
    print("Using Fashion")
    train_data = dset.FashionMNIST(root=dataset_location, train=True, download=True, transform=train_transform)
  elif dataset_name == 'svhn':
    print("Using SVHN")
    train_data = dset.SVHN(root=dataset_location, split='train', download=True, transform=train_transform)
  elif dataset_name == 'stl10':
    print("Using STL10")
    train_data = dset.STL10(root=dataset_location, split='train', download=True, transform=train_transform)
  elif dataset_name == 'devanagari':
    print("Using DEVANAGARI")
    def grey_pil_loader(path):
      # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
      with open(path, 'rb') as f:
          img = Image.open(f)
          img = img.convert('L')
          return img
    # Ensure dataset is present in the directory args.data. Does not support auto download
    train_data = dset.ImageFolder(root=dataset_location, transform=train_transform, loader = grey_pil_loader)
  elif dataset_name == 'stacking':
      # Support for costar block stacking generator
      # sites.google.com/costardataset
      # https://github.com/ahundt/costar_dataset
      # https://sites.google.com/site/costardataset
      print("dataset----------------------", self.dataset)
      Dataset = tf.data.Dataset
      flags = tf.app.flags
      FLAGS = flags.FLAGS
      np.random.seed(0)
      val_test_size = self.valid_set_size
      if images["path"] != "":
          print("datadir------------", images["path"])
          file_names = glob.glob(os.path.expanduser(images["path"]))
          train_data = file_names[val_test_size*2:]
          validation_data = file_names[val_test_size:val_test_size*2]
          self.validation_data = validation_data
          test_data = file_names[:val_test_size]
      else:
          print("-------Loading train-test-val from txt files-------")
          self.data_base_path = os.path.expanduser(self.data_base_path)
          with open(self.data_base_path + 'costar_block_stacking_v0.3_success_only_train_files.txt', mode='r') as myfile:
              train_data = myfile.read().splitlines()
          with open(self.data_base_path + 'costar_block_stacking_v0.3_success_only_test_files.txt', mode='r') as myfile:
              test_data = myfile.read().splitlines()
          with open(self.data_base_path + 'costar_block_stacking_v0.3_success_only_val_files.txt', mode='r') as myfile:
              validation_data = myfile.read().splitlines()
          print(train_data)
          # train_data = [self.data_base_path + name for name in train_data]
          # test_data = [self.data_base_path + name for name in test_data]
          # validation_data = [self.data_base_path + name for name in validation_data]
          print(validation_data)
      # number of images to look at per example
      # TODO(ahundt) currently there is a bug in one of these calculations, lowering images per example to reduce number of steps per epoch for now.
      estimated_images_per_example = 2
      print("valid set size", val_test_size)
      # TODO(ahundt) fix quick hack to proceed through epochs faster
      # self.num_train_examples = len(train_data) * self.batch_size * estimated_images_per_example
      # self.num_train_batches = (self.num_train_examples + self.batch_size - 1) // self.batch_size
      self.num_train_examples = len(train_data) * estimated_images_per_example
      self.num_train_batches = (self.num_train_examples + self.batch_size - 1) // self.batch_size
      # output_shape = (32, 32, 3)
      # WARNING: IF YOU ARE EDITING THIS CODE, MAKE SURE TO ALSO CHECK micro_controller.py and micro_child.py WHICH ALSO HAS A GENERATOR
      if self.translation_only is True:
          # We've found evidence (but not concluded finally) in hyperopt
          # that input of the rotation component actually
          # lowers translation accuracy at least in the colored block case
          # switch between the two commented lines to go back to the prvious behavior
          # data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_15']
          # self.data_features_len = 15
          data_features = ['image_0_image_n_vec_xyz_nxygrid_12']
          self.data_features_len = 12
          label_features = ['grasp_goal_xyz_3']
          self.num_classes = 3
      elif self.rotation_only is True:
          data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_15']
          self.data_features_len = 15
          # disabled 2 lines below below because best run 2018_12_2054 was with settings above
          # include a normalized xy grid, similar to uber's coordconv
          # data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17']
          # self.data_features_len = 17
          label_features = ['grasp_goal_aaxyz_nsc_5']
          self.num_classes = 5
      elif self.stacking_reward is True:
          data_features = ['image_0_image_n_vec_0_vec_n_xyz_aaxyz_nsc_nxygrid_25']
          self.data_features_len = 25
          label_features = ['stacking_reward']
          self.num_classes = 1
      # elif self.use_root is True:
      #     data_features = ['current_xyz_aaxyz_nsc_8']
      #     self.data_features_len = 8
      #     label_features = ['grasp_goal_xyz_3']
      #     self.num_classes = 8
      else:
          # original input block
          # data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_15']
          # include a normalized xy grid, similar to uber's coordconv
          data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_nxygrid_17']
          self.data_features_len = 17
          label_features = ['grasp_goal_xyz_aaxyz_nsc_8']
          self.num_classes = 8
      if self.one_hot_encoding:
          self.data_features_len += 40
      training_generator = CostarBlockStackingSequence(
          train_data, batch_size=batch_size, verbose=0,
          label_features_to_extract=label_features,
          data_features_to_extract=data_features, output_shape=self.image_shape, shuffle=True,
          random_augmentation=self.random_augmentation, one_hot_encoding=self.one_hot_encoding)

      train_enqueuer = OrderedEnqueuer(
          training_generator,
          use_multiprocessing=False,
          shuffle=True)
      train_enqueuer.start(workers=10, max_queue_size=100)

      def train_generator(): return iter(train_enqueuer.get())

      train_dataset = Dataset.from_generator(train_generator, (tf.float32, tf.float32), (tf.TensorShape(
          [None, self.image_shape[0], self.image_shape[1], self.data_features_len]), tf.TensorShape([None, None])))
      # if self.use_root is True:
      #     train_dataset = Dataset.from_generator(train_generator, (tf.float32, tf.float32), (tf.TensorShape(
      #         [None, 2]), tf.TensorShape([None, None])))
      trainer = train_dataset.make_one_shot_iterator()
      x_train, y_train = trainer.get_next()
      # x_train_list = []
      # x_train_list[0] = np.reshape(x_train[0][0], [-1, self.image_shape[1], self.image_shape[2], 3])
      # x_train_list[1] = np.reshape(x_train[0][1], [-1, self.image_shape[1], self.image_shape[2], 3])
      # x_train_list[2] = np.reshape(x_train[0][2],[-1, ])
      # print("x shape--------------", x_train.shape)
      print("batch--------------------------",
            self.num_train_examples, self.num_train_batches)
      print("y shape--------------", y_train.shape)
      self.x_train = x_train
      self.y_train = y_train
  else:
    assert False, "Cannot get training queue for dataset"

  num_train = len(train_data)
  indices = list(range(num_train))
  if search_architecture:
    # select the 'validation' set from the training data
    split = int(np.floor(train_proportion * num_train))
    print("Total Training size", num_train)
    print("Training set size", split)
    print("Validation set size", num_train-split)
    valid_data = train_data
  else:
    split = num_train
    # get the actual train/test set
    if dataset_name == 'cifar10':
        print("Using CIFAR10")
        valid_data = dset.CIFAR10(root=dataset_location, train=search_architecture, download=True, transform=train_transform)
    elif dataset_name == 'mnist':
        print("Using MNIST")
        valid_data = dset.MNIST(root=dataset_location, train=search_architecture, download=True, transform=train_transform)
    elif dataset_name == 'emnist':
        print("Using EMNIST")
        valid_data = dset.EMNIST(root=dataset_location, split='balanced', train=search_architecture, download=True, transform=train_transform)
    elif dataset_name == 'fashion':
        print("Using Fashion")
        valid_data = dset.FashionMNIST(root=dataset_location, train=search_architecture, download=True, transform=train_transform)
    elif dataset_name == 'svhn':
        print("Using SVHN")
        valid_data = dset.SVHN(root=dataset_location, split='test', download=True, transform=train_transform)
    elif dataset_name == 'stl10':
        print("Using STL10")
        valid_data = dset.STL10(root=dataset_location, split='test', download=True, transform=train_transform)
    elif dataset_name == 'devanagari':
        print("Using DEVANAGARI")
        def grey_pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
          with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            return img
        # Ensure dataset is present in the directory args.data. Does not support auto download
        valid_data = dset.ImageFolder(root=dataset_location, transform=train_transform, loader = grey_pil_loader)
    else:
        assert False, "Cannot get training queue for dataset"

  if dataset_name == 'devanagari':
    print("SHUFFLE INDEX LIST BEFORE BATCHING")
    print("Before Shuffle", indices[-10:num_train])
    np.random.shuffle(indices)
    print("After Shuffle", indices[-10:num_train])

  # shuffle does not need to be set to True because
  # that is taken care of by the subset random sampler
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=4)

  if search_architecture:
    # validation sampled from training set
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=4)
  else:
    # test set
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        pin_memory=False, num_workers=4)

  return train_queue, valid_queue
