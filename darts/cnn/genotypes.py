from collections import namedtuple

Genotype = namedtuple('Genotype', 'start start_concat normal normal_concat reduce reduce_concat end end_concat aux')

# Simplified new version based on actual results
# TODO(ahundt) enable different primitives and reduce primitives
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    # 'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    # 'sep_conv_5x5',
    'dil_conv_3x3',
    # 'dil_conv_5x5',
    # 'nor_conv_3x3',
    # 'nor_conv_5x5',
    # 'nor_conv_7x7',
    'flood_conv_3x3',
    'dil_flood_conv_3x3',
    'choke_conv_3x3',
    'dil_choke_conv_3x3',
]

REDUCE_PRIMITIVES = [
    'none',
    'max_pool_3x3',
    # 'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    # 'sep_conv_5x5',
    'dil_conv_3x3',
    # 'dil_conv_5x5',
    # 'nor_conv_3x3',
    # 'nor_conv_5x5',
    # 'nor_conv_7x7',
    'flood_conv_3x3',
    'dil_flood_conv_3x3',
    'choke_conv_3x3',
    'dil_choke_conv_3x3',
]
''' Old Version
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    # 'nor_conv_3x3',
    # 'nor_conv_5x5',
    # 'nor_conv_7x7',
]
'''

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],
)

AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6],
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],
)

# source: https://github.com/chenxi116/PNASNet.pytorch
PNASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 1),
    ('max_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 4),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 1),
    ('max_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 4),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('skip_connect', 1),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],
)

# search-20181213-172120-choke_flood_simple_mixed_aux_v2-cifar10
CHOKE_FLOOD = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('dil_flood_conv_3x3', 2), ('skip_connect', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 1), ('dil_flood_conv_3x3', 2), ('dil_choke_conv_3x3', 3)],
  reduce_concat=range(2, 6),
  start = [],
  start_concat = [],
  end = [],
  end_concat = [], aux=[0.019386334344744682, 0.021869726479053497, 0.019505096599459648, 0.020072733983397484, 0.02058151178061962, 0.20746095478534698, 0.43517687916755676, 0.2559467852115631])


# 2018_12_27_19_54_50
# SEARCH BEST EPOCH: 2018_12_27_20_48_26 epoch, 104, train_acc, 99.053333, valid_acc, 93.960000, train_loss, 0.028875, valid_loss, 0.220934, lr, 1.081639e-03, best_epoch, 104, best_valid_acc, 93.960000
# Experiment search dir : search-20181223-232449-choke_flood_simple_start_end_cells_cutout_mixed_aux-cifar10
# SEARCH time and epochs: 120/120 [106:46:51<00:00, 3213.34s/it]
CHOKE_FLOOD_START_END = Genotype(
start=[('flood_conv_3x3', 0), ('flood_conv_3x3', 1), ('choke_conv_3x3', 0), ('choke_conv_3x3', 1), ('flood_conv_3x3', 0), ('choke_conv_3x3', 2), ('flood_conv_3x3', 3), ('sep_conv_3x3', 4)], start_concat=range(2, 6),
normal=[('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('flood_conv_3x3', 0), ('dil_flood_conv_3x3', 2)], reduce_concat=range(2, 6),
end=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('skip_connect', 3)], end_concat=range(2, 6),
aux=[0.037209805101156235, 0.0371854193508625, 0.03715495020151138, 0.03715408220887184, 0.03715182840824127, 0.037225864827632904, 0.20869530737400055, 0.5682227611541748])

# really long really slow second derivative training run:
# search-20181211-142658-choke_flood_primitives-cifar10
# 2018_12_21_10_55_01 valid_acc 92.640000 2018_12_21_10_54_34 train_acc 99.628889 2018_12_21_10_55_01 epoch 85 lr 5.544224e-03
HESSIAN_CHOKE_FLOOD = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('flood_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 3), ('dil_choke_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('flood_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('choke_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6),
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],)

FASHION = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5],
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5],
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],)
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5],
  start = [],
  start_concat = [],
  end = [],
  end_concat = [],
  aux = [],)

DARTS = DARTS_V2

