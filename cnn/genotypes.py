from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Primitives for the dilation, sep_conv, flood, and choke 3x3 only search space
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

# Primitives for the original darts search space
DARTS_PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

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
  reduce_concat = [3, 4, 6]
)

'''
2019_01_13_23_24_09 args = Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=32, cutout=False, cutout_length=16, data='../data', drop_path_prob=0.3, epochs=60, gpu=0, grad_clip=5, init_channels=16, layers=8, learning_rate=0.025, learning_rate_min=0.001, model_path='saved_models', momentum=0.9, random_eraser=False, report_freq=50, save='search-choke_flood_45b2033_branch_merge_mixed_aux-20190113-232409', seed=2, train_portion=0.5, unrolled=False, weight_decay=0.0003)
2019_01_13_23_24_13 param size = 5.867642MB

2019_01_15_18_59_41 epoch, 56, train_acc, 99.852000, valid_acc, 91.428000, train_loss, 0.013050, valid_loss, 0.289964, lr, 1.262229e-03, best_epoch, 56, best_valid_acc, 91.428000
2019_01_15_18_59_41 genotype = Genotype(normal=[('choke_conv_3x3', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0)
, ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect
', 2), ('flood_conv_3x3', 0)], reduce_concat=range(2, 6))
2019_01_15_18_59_41 alphas_normal = tensor([[0.1227, 0.0541, 0.1552, 0.1318, 0.0737, 0.1543, 0.0597, 0.1619, 0.0866],
        [0.5278, 0.0389, 0.0549, 0.0694, 0.0384, 0.0499, 0.0367, 0.1205, 0.0634],
        [0.3518, 0.0780, 0.2044, 0.0623, 0.0452, 0.0943, 0.0437, 0.0722, 0.0482],
        [0.7436, 0.0338, 0.0540, 0.0221, 0.0204, 0.0241, 0.0178, 0.0555, 0.0289],
        [0.8558, 0.0148, 0.0292, 0.0198, 0.0113, 0.0152, 0.0116, 0.0251, 0.0172],
        [0.6509, 0.0425, 0.0983, 0.0329, 0.0262, 0.0426, 0.0209, 0.0483, 0.0376],
        [0.8584, 0.0184, 0.0246, 0.0181, 0.0129, 0.0195, 0.0110, 0.0209, 0.0161],
        [0.8986, 0.0105, 0.0198, 0.0105, 0.0103, 0.0117, 0.0098, 0.0172, 0.0116],
        [0.9225, 0.0084, 0.0135, 0.0082, 0.0078, 0.0088, 0.0069, 0.0134, 0.0105],
        [0.6484, 0.0409, 0.1115, 0.0283, 0.0309, 0.0314, 0.0253, 0.0414, 0.0421],
        [0.8666, 0.0169, 0.0248, 0.0194, 0.0132, 0.0142, 0.0112, 0.0217, 0.0121],
        [0.9063, 0.0106, 0.0195, 0.0108, 0.0088, 0.0098, 0.0083, 0.0152, 0.0108],
        [0.9359, 0.0070, 0.0109, 0.0083, 0.0070, 0.0072, 0.0068, 0.0089, 0.0080],
        [0.9238, 0.0069, 0.0132, 0.0127, 0.0069, 0.0087, 0.0072, 0.0122, 0.0084]],
       device='cuda:0', grad_fn=<SoftmaxBackward>)
2019_01_15_18_59_41 alphas_reduce = tensor([[0.0785, 0.1742, 0.1547, 0.0916, 0.0640, 0.1167, 0.0835, 0.1516, 0.0852],
        [0.1119, 0.1278, 0.1660, 0.0930, 0.0867, 0.1177, 0.0962, 0.1181, 0.0825],
        [0.0724, 0.2795, 0.0808, 0.0926, 0.0741, 0.1522, 0.0712, 0.0876, 0.0895],
        [0.1023, 0.1814, 0.1192, 0.0859, 0.0959, 0.1292, 0.0849, 0.1003, 0.1009],
        [0.1553, 0.1172, 0.2175, 0.0948, 0.0801, 0.0805, 0.0803, 0.0903, 0.0841],
        [0.0763, 0.2053, 0.1521, 0.0969, 0.0904, 0.1321, 0.0729, 0.1067, 0.0674],
        [0.1005, 0.1378, 0.1372, 0.0933, 0.0854, 0.1270, 0.1148, 0.1048, 0.0992],
        [0.1385, 0.1063, 0.1778, 0.1054, 0.1055, 0.0974, 0.1002, 0.0795, 0.0894],
        [0.1626, 0.0849, 0.1274, 0.1046, 0.0919, 0.1014, 0.1248, 0.1103, 0.0921],
        [0.0761, 0.1416, 0.1477, 0.1104, 0.0809, 0.1483, 0.1008, 0.1024, 0.0917],
        [0.1175, 0.1253, 0.1357, 0.1014, 0.0903, 0.1062, 0.1185, 0.1115, 0.0935],
        [0.1564, 0.1003, 0.1710, 0.1140, 0.0702, 0.1052, 0.0877, 0.0980, 0.0972],
        [0.2252, 0.0806, 0.1401, 0.1061, 0.0844, 0.0950, 0.0902, 0.1027, 0.0756],
        [0.2697, 0.0890, 0.1412, 0.0795, 0.0863, 0.0738, 0.0654, 0.1088, 0.0863]],
       device='cuda:0', grad_fn=<SoftmaxBackward>)

2019_01_15_21_19_12 epoch, 59, train_acc, 99.900000, valid_acc, 91.232000, train_loss, 0.013466, valid_loss, 0.282533, lr, 1.016446e-03, best_epoch, 56, best_valid_acc, 91.428000
100%|| 60/60 [45:54:58<00:00, 2783.23s/it]
2019_01_15_21_19_12 genotype = Genotype(normal=[('choke_conv_3x3', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('flood_conv_3x3', 0)], reduce_concat=range(2, 6))
2019_01_15_21_19_12 Search for Model Complete! Save dir: search-choke_flood_45b2033_branch_merge_mixed_aux-20190113-232409
'''
CHOKE_FLOOD_DIL_IS_SEP_CONV = Genotype(normal=[('choke_conv_3x3', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('flood_conv_3x3', 0)], reduce_concat=range(2, 6))

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

