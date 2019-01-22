# AutoAugment policies found on Cifar.
# Paper: https://arxiv.org/abs/1805.09501
# Code: https://github.com/DeepVoltaire/AutoAugment
# License: MIT
# Code: https://github.com/tensorflow/models/blob/903194c51d4798df25334dd5ccecc2604974efd9/research/autoaugment/policies.py
# license: Apache v2
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


def create_cutout_mask(img_height, img_width, num_channels, size):
  """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.
  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
  assert img_height == img_width

  # Sample center where cutout mask will be applied
  height_loc = np.random.randint(low=0, high=img_height)
  width_loc = np.random.randint(low=0, high=img_width)

  # Determine upper right and lower left corners of patch
  upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
  lower_coord = (min(img_height, height_loc + size // 2),
                 min(img_width, width_loc + size // 2))
  mask_height = lower_coord[0] - upper_coord[0]
  mask_width = lower_coord[1] - upper_coord[1]
  assert mask_height > 0
  assert mask_width > 0

  mask = np.ones((img_height, img_width, num_channels))
  zeros = np.zeros((mask_height, mask_width, num_channels))
  mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
      zeros)
  return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
  """Apply cutout with mask of shape `size` x `size` to `img`.
  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.
  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be
  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
  img_height, img_width, num_channels = (img.shape[0], img.shape[1],
                                         img.shape[2])
  assert len(img.shape) == 3
  mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
  return img * mask


def int_parameter(level, maxval, parameter_max=10):
  """Helper function to scale `val` between 0 and maxval.

  source: https://github.com/tensorflow/models/blob/903194c51d4798df25334dd5ccecc2604974efd9/research/autoaugment/augmentation_transforms.py

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / parameter_max)


def _cutout_pil_impl(pil_img, level):
  """Apply cutout to pil_img at the specified level.
  """
  size = int_parameter(level, 20)
  if size <= 0:
    return pil_img
  img_height, img_width, num_channels = (32, 32, 3)
  _, upper_coord, lower_coord = (
      create_cutout_mask(img_height, img_width, num_channels, size))
  pixels = pil_img.load()  # create the pixel map
  for i in range(upper_coord[0], lower_coord[1]):  # for every col:
    for j in range(upper_coord[0], lower_coord[1]):  # For every row
      pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
  return pil_img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        # AutoAugment policies found on Cifar.
        # source: https://github.com/tensorflow/models/blob/903194c51d4798df25334dd5ccecc2604974efd9/research/autoaugment/policies.py
        # license: Apache v2
        # Paper: https://arxiv.org/abs/1805.09501
        exp0_0            =    [
          SubPolicy(0.1, 'invert',       7,    0.2, 'contrast',       6, fillcolor),
          SubPolicy(0.7, 'rotate',       2,    0.3, 'translateX',     9, fillcolor),
          SubPolicy(0.8, 'sharpness',    1,    0.9, 'sharpness',      3, fillcolor),
          SubPolicy(0.5, 'shearY',       8,    0.7, 'translateY',     9, fillcolor),
          SubPolicy(0.5, 'autocontrast', 8,    0.9, 'equalize',       2, fillcolor),]
        exp0_1            =    [
          SubPolicy(0.4, 'solarize',     5,    0.9, 'autocontrast',   3, fillcolor),
          SubPolicy(0.9, 'translateY',   9,    0.7, 'translateY',     9, fillcolor),
          SubPolicy(0.9, 'autocontrast', 2,    0.8, 'solarize',       3, fillcolor),
          SubPolicy(0.8, 'equalize',     8,    0.1, 'invert',         3, fillcolor),
          SubPolicy(0.7, 'translateY',   9,    0.9, 'autocontrast',   1, fillcolor),]
        exp0_2            =    [
          SubPolicy(0.4, 'solarize',     5,    0.0, 'autocontrast',   2, fillcolor),
          SubPolicy(0.7, 'translateY',   9,    0.7, 'translateY',     9, fillcolor),
          SubPolicy(0.9, 'autocontrast', 0,    0.4, 'solarize',       3, fillcolor),
          SubPolicy(0.7, 'equalize',     5,    0.1, 'invert',         3, fillcolor),
          SubPolicy(0.7, 'translateY',   9,    0.7, 'translateY',     9, fillcolor),]
        exp0_3            =    [
          SubPolicy(0.4, 'solarize',     5,    0.9, 'autocontrast',   1, fillcolor),
          SubPolicy(0.8, 'translateY',   9,    0.9, 'translateY',     9, fillcolor),
          SubPolicy(0.8, 'autocontrast', 0,    0.7, 'translateY',     9, fillcolor),
          SubPolicy(0.2, 'translateY',   7,    0.9, 'color',          6, fillcolor),
          SubPolicy(0.7, 'equalize',     6,    0.4, 'color',          9, fillcolor),]
        exp1_0            =    [
          SubPolicy(0.2, 'shearY',       7,    0.3, 'posterize',      7, fillcolor),
          SubPolicy(0.4, 'color',        3,    0.6, 'brightness',     7, fillcolor),
          SubPolicy(0.3, 'sharpness',    9,    0.7, 'brightness',     9, fillcolor),
          SubPolicy(0.6, 'equalize',     5,    0.5, 'equalize',       1, fillcolor),
          SubPolicy(0.6, 'contrast',     7,    0.6, 'sharpness',      5, fillcolor),]
        exp1_1            =    [
          SubPolicy(0.3, 'brightness',   7,    0.5, 'autocontrast',   8, fillcolor),
          SubPolicy(0.9, 'autocontrast', 4,    0.5, 'autocontrast',   6, fillcolor),
          SubPolicy(0.3, 'solarize',     5,    0.6, 'equalize',       5, fillcolor),
          SubPolicy(0.2, 'translateY',   4,    0.3, 'sharpness',      3, fillcolor),
          SubPolicy(0.0, 'brightness',   8,    0.8, 'color',          8, fillcolor),]
        exp1_2            =    [
          SubPolicy(0.2, 'solarize',     6,    0.8, 'color',          6, fillcolor),
          SubPolicy(0.2, 'solarize',     6,    0.8, 'autocontrast',   1, fillcolor),
          SubPolicy(0.4, 'solarize',     1,    0.6, 'equalize',       5, fillcolor),
          SubPolicy(0.0, 'brightness',   0,    0.5, 'solarize',       2, fillcolor),
          SubPolicy(0.9, 'autocontrast', 5,    0.5, 'brightness',     3, fillcolor),]
        exp1_3            =    [
          SubPolicy(0.7, 'contrast',     5,    0.0, 'brightness',     2, fillcolor),
          SubPolicy(0.2, 'solarize',     8,    0.1, 'solarize',       5, fillcolor),
          SubPolicy(0.5, 'contrast',     1,    0.2, 'translateY',     9, fillcolor),
          SubPolicy(0.6, 'autocontrast', 5,    0.0, 'translateY',     9, fillcolor),
          SubPolicy(0.9, 'autocontrast', 4,    0.8, 'equalize',       4, fillcolor),]
        exp1_4            =    [
          SubPolicy(0.0, 'brightness',   7,    0.4, 'equalize',       7, fillcolor),
          SubPolicy(0.2, 'solarize',     5,    0.7, 'equalize',       5, fillcolor),
          SubPolicy(0.6, 'equalize',     8,    0.6, 'color',          2, fillcolor),
          SubPolicy(0.3, 'color',        7,    0.2, 'color',          4, fillcolor),
          SubPolicy(0.5, 'autocontrast', 2,    0.7, 'solarize',       2, fillcolor),]
        exp1_5            =    [
          SubPolicy(0.2, 'autocontrast', 0,    0.1, 'equalize',       0, fillcolor),
          SubPolicy(0.6, 'shearY',       5,    0.6, 'equalize',       5, fillcolor),
          SubPolicy(0.9, 'brightness',   3,    0.4, 'autocontrast',   1, fillcolor),
          SubPolicy(0.8, 'equalize',     8,    0.7, 'equalize',       7, fillcolor),
          SubPolicy(0.7, 'equalize',     7,    0.5, 'solarize',       0, fillcolor),]
        exp1_6            =    [
          SubPolicy(0.8, 'equalize',     4,    0.8, 'translateY',     9, fillcolor),
          SubPolicy(0.8, 'translateY',   9,    0.6, 'translateY',     9, fillcolor),
          SubPolicy(0.9, 'translateY',   0,    0.5, 'translateY',     9, fillcolor),
          SubPolicy(0.5, 'autocontrast', 3,    0.3, 'solarize',       4, fillcolor),
          SubPolicy(0.5, 'solarize',     3,    0.4, 'equalize',       4, fillcolor),]
        exp2_0            =    [
          SubPolicy(0.7, 'color',        7,    0.5, 'translateX',     8, fillcolor),
          SubPolicy(0.3, 'equalize',     7,    0.4, 'autocontrast',   8, fillcolor),
          SubPolicy(0.4, 'translateY',   3,    0.2, 'sharpness',      6, fillcolor),
          SubPolicy(0.9, 'brightness',   6,    0.2, 'color',          8, fillcolor),
          SubPolicy(0.5, 'solarize',     2,    0.0, 'invert',         3, fillcolor),]
        exp2_1            =    [
          SubPolicy(0.1, 'autocontrast', 5,    0.0, 'brightness',     0, fillcolor),
          SubPolicy(0.2, 'cutout',       4,    0.1, 'equalize',       1, fillcolor),
          SubPolicy(0.7, 'equalize',     7,    0.6, 'autocontrast',   4, fillcolor),
          SubPolicy(0.1, 'color',        8,    0.2, 'shearY',         3, fillcolor),
          SubPolicy(0.4, 'shearY',       2,    0.7, 'rotate',         0, fillcolor),]
        exp2_2            =    [
          SubPolicy(0.1, 'shearY',       3,    0.9, 'autocontrast',   5, fillcolor),
          SubPolicy(0.3, 'translateY',   6,    0.3, 'cutout',         3, fillcolor),
          SubPolicy(0.5, 'equalize',     0,    0.6, 'solarize',       6, fillcolor),
          SubPolicy(0.3, 'autocontrast', 5,    0.2, 'rotate',         7, fillcolor),
          SubPolicy(0.8, 'equalize',     2,    0.4, 'invert',         0, fillcolor),]
        exp2_3            =    [
          SubPolicy(0.9, 'equalize',     5,    0.7, 'color',          0, fillcolor),
          SubPolicy(0.1, 'equalize',     1,    0.1, 'shearY',         3, fillcolor),
          SubPolicy(0.7, 'autocontrast', 3,    0.7, 'equalize',       0, fillcolor),
          SubPolicy(0.5, 'brightness',   1,    0.1, 'contrast',       7, fillcolor),
          SubPolicy(0.1, 'contrast',     4,    0.6, 'solarize',       5, fillcolor),]
        exp2_4            =    [
          SubPolicy(0.2, 'solarize',     3,    0.0, 'shearX',         0, fillcolor),
          SubPolicy(0.3, 'translateX',   0,    0.6, 'translateX',     0, fillcolor),
          SubPolicy(0.5, 'equalize',     9,    0.6, 'translateY',     7, fillcolor),
          SubPolicy(0.1, 'shearX',       0,    0.5, 'sharpness',      1, fillcolor),
          SubPolicy(0.8, 'equalize',     6,    0.3, 'invert',         6, fillcolor),]
        exp2_5            =    [
          SubPolicy(0.3, 'autocontrast', 9,    0.5, 'cutout',         3, fillcolor),
          SubPolicy(0.4, 'shearX',       4,    0.9, 'autocontrast',   2, fillcolor),
          SubPolicy(0.0, 'shearX',       3,    0.0, 'posterize',      3, fillcolor),
          SubPolicy(0.4, 'solarize',     3,    0.2, 'color',          4, fillcolor),
          SubPolicy(0.1, 'equalize',     4,    0.7, 'equalize',       6, fillcolor),]
        exp2_6            =    [
          SubPolicy(0.3, 'equalize',     8,    0.4, 'autocontrast',   3, fillcolor),
          SubPolicy(0.6, 'solarize',     4,    0.7, 'autocontrast',   6, fillcolor),
          SubPolicy(0.2, 'autocontrast', 9,    0.4, 'brightness',     8, fillcolor),
          SubPolicy(0.1, 'equalize',     0,    0.0, 'equalize',       6, fillcolor),
          SubPolicy(0.8, 'equalize',     4,    0.0, 'equalize',       4, fillcolor),]
        exp2_7            =    [
          SubPolicy(0.5, 'equalize',     5,    0.1, 'autocontrast',   2, fillcolor),
          SubPolicy(0.5, 'solarize',     5,    0.9, 'autocontrast',   5, fillcolor),
          SubPolicy(0.6, 'autocontrast', 1,    0.7, 'autocontrast',   8, fillcolor),
          SubPolicy(0.2, 'equalize',     0,    0.1, 'autocontrast',   2, fillcolor),
          SubPolicy(0.6, 'equalize',     9,    0.4, 'equalize',       4, fillcolor),]
        exp0s             =  exp0_0 + exp0_1 + exp0_2 + exp0_3
        exp1s             =  exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
        exp2s             =  exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
        self.policies = exp0s + exp1s + exp2s

    def __call__(self, img):
        # Choose policy then sub-policy. Note in the original paper the policy is chosen once per epoch.
        policy = self.policies[np.random.choice(len(self.policies))]
        sub_policy = policy[np.random.choice(len(policy))]
        return sub_policy(img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "cutout": np.linspace(0.0, 0.9, 10),
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            # TODO(ahundt) cutout is a no-op until this pull request is updated https://github.com/tensorflow/models/pull/6078
            "cutout": lambda img, magnitude: img
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1,
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img