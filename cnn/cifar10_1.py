# Source: https://github.com/kharvd/cifar-10.1-pytorch
# License: MIT
import io
import os
import os.path
import pickle

import numpy as np

from PIL import Image

import torch.utils.data as data

from torchvision.datasets.utils import download_url, check_integrity

def load_new_test_data(root, version='default'):
    data_path = root
    filename = 'cifar10.1'
    if version == 'default':
        pass
    elif version == 'v0':
        filename += '-v0'
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version))
    label_filename = filename + '-labels.npy'
    imagedata_filename = filename + '-data.npy'
    label_filepath = os.path.join(data_path, label_filename)
    imagedata_filepath = os.path.join(data_path, imagedata_filename)
    labels = np.load(label_filepath).astype(np.int64)
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version == 'default':
        assert labels.shape[0] == 2000
    elif version == 'v0':
        assert labels.shape[0] == 2021
    return imagedata, labels


class CIFAR10_1(data.Dataset):
    images_url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy'
    images_md5 = '29615bb88ff99bca6b147cee2520f010'
    images_filename = 'cifar10.1-data.npy'

    labels_url = 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy'
    labels_md5 = 'a27460fa134ae91e4a5cb7e6be8d269e'
    labels_filename = 'cifar10.1-labels.npy'

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ]

    @property
    def targets(self):
        return self.labels

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        images, labels = load_new_test_data(root)

        self.data = images
        self.labels = labels

        self.class_to_idx = {
            _class: i
            for i, _class in enumerate(self.classes)
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        data_path = os.path.join(self.root, self.images_filename)
        labels_path = os.path.join(self.root, self.labels_filename)
        return (check_integrity(data_path, self.images_md5) and
            check_integrity(labels_path, self.labels_md5))

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.images_url, root, self.images_filename, self.images_md5)
        download_url(self.labels_url, root, self.labels_filename, self.labels_md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str