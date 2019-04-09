from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat layout')

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
  layout='cell',
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
  layout='cell',
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

2019_01_21_23_21_59 gpu device = 0
2019_01_21_23_21_59 args = Namespace(arch='DARTS', autoaugment=True, auxiliary=True, auxiliary_weight=0.4, batch_size=64, cutout=True, cutout_length=16, data='../data', dataset='cifar10', drop_path
_prob=0.2, epochs=1000, gpu=0, grad_clip=5, init_channels=36, layers=20, learning_rate=0.025, learning_rate_min=1e-07, mixed_auxiliary=False, model_path='saved_models', momentum=0.9, ops='OPS', opt
imizer='sgd', partial=0.125, primitives='PRIMITIVES', random_eraser=False, report_freq=50, save='eval-20190121-232159-AUTOAUGMENT_V2_KEY_PADDING_d5dda02_BUGFIX-cifar10-DARTS', seed=4, warm_restarts
=20, weight_decay=0.0003)
2019_01_21_23_22_02 param size = 3.529270MB
2019_01_25_20_26_22 best_epoch, 988, best_train_acc, 95.852000, best_valid_acc, 97.890000, best_train_loss, 0.196667, best_valid_loss, 0.076396, lr, 8.881592e-06, best_epoch, 988, best_valid_acc, 97.890000 cifar10.1_valid_acc, 93.750000, cifar10.1_valid_loss, 0.218554
2019_01_25_20_26_22 Training of Final Model Complete! Save dir: eval-20190121-232159-AUTOAUGMENT_V2_KEY_PADDING_d5dda02_BUGFIX-cifar10-DARTS
'''
CHOKE_FLOOD_DIL_IS_SEP_CONV = Genotype(normal=[('choke_conv_3x3', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('flood_conv_3x3', 0)], reduce_concat=range(2, 6), layout='cell')
SHARP_DARTS = CHOKE_FLOOD_DIL_IS_SEP_CONV

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5], layout='cell')
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5], layout='cell')

DARTS = DARTS_V2

"""
Save dir: search-SEARCH_REPRODUCTION_ATTEMPT_KEY_PADDING_56b8fe9_BUGFIX_Dil_is_SepConv-20190113-231854
2019_01_15_02_35_16 epoch, 50, train_acc, 99.592000, valid_acc, 90.892000, train_loss, 0.019530, valid_loss, 0.330776, lr, 2.607695e-03, best_epoch, 50, best_valid_acc, 90.892000
2019_01_15_02_35_16 genotype = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('se
p_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
 ('skip_connect', 3)], reduce_concat=range(2, 6))
2019_01_15_02_35_16 alphas_normal = tensor([[0.1797, 0.0438, 0.0386, 0.0838, 0.3695, 0.1124, 0.1174, 0.0549],
        [0.4426, 0.0311, 0.0249, 0.0465, 0.1600, 0.1146, 0.1076, 0.0728],
        [0.5580, 0.0543, 0.0353, 0.0886, 0.1094, 0.0682, 0.0497, 0.0365],
        [0.6387, 0.0380, 0.0251, 0.0598, 0.0863, 0.0605, 0.0545, 0.0372],
        [0.7500, 0.0194, 0.0159, 0.0430, 0.0884, 0.0320, 0.0283, 0.0231],
        [0.6430, 0.0518, 0.0424, 0.0960, 0.0508, 0.0493, 0.0349, 0.0317],
        [0.7367, 0.0274, 0.0203, 0.0383, 0.0779, 0.0358, 0.0367, 0.0268],
        [0.8526, 0.0150, 0.0127, 0.0276, 0.0364, 0.0201, 0.0166, 0.0190],
        [0.9045, 0.0083, 0.0082, 0.0125, 0.0259, 0.0170, 0.0121, 0.0114],
        [0.7205, 0.0538, 0.0398, 0.0811, 0.0352, 0.0233, 0.0218, 0.0244],
        [0.6869, 0.0426, 0.0296, 0.0604, 0.0792, 0.0390, 0.0337, 0.0285],
        [0.9148, 0.0113, 0.0098, 0.0176, 0.0143, 0.0106, 0.0108, 0.0109],
        [0.9354, 0.0068, 0.0069, 0.0102, 0.0117, 0.0100, 0.0098, 0.0090],
        [0.9294, 0.0068, 0.0073, 0.0112, 0.0125, 0.0120, 0.0106, 0.0101]],
       device='cuda:0', grad_fn=<SoftmaxBackward>)
2019_01_15_02_35_16 alphas_reduce = tensor([[0.0757, 0.2186, 0.2142, 0.1140, 0.1221, 0.1052, 0.0843, 0.0659],
        [0.1503, 0.1184, 0.1191, 0.1474, 0.1392, 0.1070, 0.1051, 0.1135],
        [0.0742, 0.2551, 0.2631, 0.0914, 0.0927, 0.0778, 0.0800, 0.0656],
        [0.1337, 0.1702, 0.1886, 0.1128, 0.0885, 0.1077, 0.1070, 0.0915],
        [0.1277, 0.0884, 0.1102, 0.3171, 0.1124, 0.0774, 0.0781, 0.0886],
        [0.0838, 0.1910, 0.2192, 0.1022, 0.0987, 0.1073, 0.1029, 0.0949],
        [0.1147, 0.1692, 0.2006, 0.1156, 0.1039, 0.1055, 0.1026, 0.0879],
        [0.1195, 0.0778, 0.1036, 0.2572, 0.1311, 0.0998, 0.0992, 0.1117],
        [0.2289, 0.0652, 0.0827, 0.2181, 0.1189, 0.0921, 0.0883, 0.1059],
        [0.0807, 0.1987, 0.2584, 0.1142, 0.1083, 0.0770, 0.0872, 0.0754],
        [0.1019, 0.1489, 0.1791, 0.2150, 0.0844, 0.0986, 0.0964, 0.0757],
        [0.1322, 0.0671, 0.1000, 0.3702, 0.0891, 0.0758, 0.0937, 0.0718],
        [0.2760, 0.0609, 0.0833, 0.2941, 0.0900, 0.0576, 0.0620, 0.0761],
        [0.3477, 0.0537, 0.0752, 0.1774, 0.1061, 0.0828, 0.0791, 0.0781]],
       device='cuda:0', grad_fn=<SoftmaxBackward>)
2019_01_15_06_53_06 alphas_normal = tensor([[0.2425, 0.0422, 0.0396, 0.0830, 0.3289, 0.1043, 0.1095, 0.0501],
        [0.5780, 0.0277, 0.0235, 0.0404, 0.1082, 0.0867, 0.0774, 0.0581],
        [0.6560, 0.0415, 0.0281, 0.0678, 0.0829, 0.0519, 0.0427, 0.0292],
        [0.7764, 0.0247, 0.0177, 0.0391, 0.0483, 0.0376, 0.0328, 0.0235],
        [0.8400, 0.0133, 0.0113, 0.0259, 0.0547, 0.0203, 0.0193, 0.0153],
        [0.7062, 0.0414, 0.0372, 0.0807, 0.0400, 0.0407, 0.0283, 0.0255],
        [0.8420, 0.0182, 0.0142, 0.0242, 0.0447, 0.0211, 0.0202, 0.0154],
        [0.8965, 0.0113, 0.0101, 0.0180, 0.0245, 0.0140, 0.0120, 0.0136],
        [0.9272, 0.0072, 0.0072, 0.0121, 0.0173, 0.0115, 0.0092, 0.0083],
        [0.7692, 0.0434, 0.0354, 0.0685, 0.0301, 0.0175, 0.0168, 0.0192],
        [0.7816, 0.0323, 0.0235, 0.0458, 0.0513, 0.0253, 0.0220, 0.0183],
        [0.9317, 0.0093, 0.0083, 0.0133, 0.0112, 0.0086, 0.0087, 0.0088],
        [0.9445, 0.0063, 0.0066, 0.0103, 0.0095, 0.0078, 0.0082, 0.0068],
        [0.9430, 0.0064, 0.0069, 0.0115, 0.0084, 0.0087, 0.0076, 0.0076]],
       device='cuda:0', grad_fn=<SoftmaxBackward>)
2019_01_15_06_53_06 alphas_reduce = tensor([[0.0770, 0.2144, 0.2315, 0.1136, 0.1204, 0.0999, 0.0813, 0.0619],
        [0.1485, 0.1163, 0.1218, 0.1499, 0.1427, 0.1029, 0.1034, 0.1145],
        [0.0747, 0.2417, 0.2707, 0.0923, 0.0982, 0.0748, 0.0807, 0.0669],
        [0.1287, 0.1730, 0.2038, 0.1092, 0.0911, 0.0996, 0.1101, 0.0845],
        [0.1265, 0.0857, 0.1139, 0.3273, 0.1120, 0.0762, 0.0742, 0.0841],
        [0.0849, 0.1803, 0.2166, 0.1040, 0.0994, 0.1073, 0.1037, 0.1039],
        [0.1143, 0.1651, 0.2031, 0.1176, 0.1033, 0.1082, 0.1031, 0.0854],
        [0.1198, 0.0768, 0.1047, 0.2503, 0.1362, 0.1031, 0.1022, 0.1070],
        [0.2233, 0.0662, 0.0858, 0.2163, 0.1215, 0.0968, 0.0879, 0.1023],
        [0.0837, 0.1847, 0.2635, 0.1174, 0.1096, 0.0781, 0.0877, 0.0753],
        [0.1021, 0.1434, 0.1805, 0.2217, 0.0802, 0.0952, 0.1003, 0.0767],
        [0.1314, 0.0662, 0.0994, 0.3936, 0.0815, 0.0713, 0.0887, 0.0680],
        [0.2715, 0.0612, 0.0849, 0.3124, 0.0838, 0.0562, 0.0598, 0.0703],
        [0.3690, 0.0554, 0.0788, 0.1813, 0.0963, 0.0740, 0.0747, 0.0705]],
       device='cuda:0', grad_fn=<SoftmaxBackward>)
2019_01_15_07_25_08 epoch, 59, train_acc, 99.752000, valid_acc, 90.732000, train_loss, 0.018408, valid_loss, 0.312794, lr, 1.016446e-03, best_epoch, 50, best_valid_acc, 90.892000
100%|| 60/60 [32:06:09<00:00, 1929.41s/it]
2019_01_15_07_25_08 genotype = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
2019_01_15_07_25_08 Search for Model Complete! Save dir: search-SEARCH_REPRODUCTION_ATTEMPT_KEY_PADDING_56b8fe9_BUGFIX_Dil_is_SepConv-20190113-231854

± export CUDA_VISIBLE_DEVICES="1" && python3 train.py --autoaugment --auxiliary --cutout --batch_size 48 --epochs 1000 --save REPRODUCTION_ATTEMPT_KEY_PADDING_`git rev-parse --short HEAD`_AUTOAUGME
NT --arch DARTS_PRIMITIVES_DIL_IS_SEPCONV
Experiment dir : eval-20190121-225901-REPRODUCTION_ATTEMPT_KEY_PADDING_2a88102_AUTOAUGMENT-cifar10-DARTS_PRIMITIVES_DIL_IS_SEPCONV
2019_01_21_22_59_01 gpu device = 0
2019_01_21_22_59_01 args = Namespace(arch='DARTS_PRIMITIVES_DIL_IS_SEPCONV', autoaugment=True, auxiliary=True, auxiliary_weight=0.4, batch_size=48, cutout=True, cutout_length=16, data='../data', da
taset='cifar10', drop_path_prob=0.2, epochs=1000, gpu=0, grad_clip=5, init_channels=36, layers=20, learning_rate=0.025, learning_rate_min=1e-07, mixed_auxiliary=False, model_path='saved_models', mo
mentum=0.9, ops='OPS', optimizer='sgd', partial=0.125, primitives='PRIMITIVES', random_eraser=False, report_freq=50, save='eval-20190121-225901-REPRODUCTION_ATTEMPT_KEY_PADDING_2a88102_AUTOAUGMENT-
cifar10-DARTS_PRIMITIVES_DIL_IS_SEPCONV', seed=0, warm_restarts=20, weight_decay=0.0003)
loading op dict: operations.OPS
loading primitives: genotypes.PRIMITIVES
Validation step: 41, loss:   0.24891, top 1: 93.40 top 5: 99.85 progress: 100%|| 42/42 [00:03<00:00, 12.56it/s]
2019_01_27_09_14_32 best_epoch, 940, best_train_acc, 94.665997, best_valid_acc, 97.519998, best_train_loss, 0.235156, best_valid_loss, 0.087909, lr, 2.214094e-04, best_epoch, 940, best_valid_acc, 97.519998 cifar10.1_valid_acc, 93.399997, cifar10.1_valid_loss, 0.248915
2019_01_27_09_14_32 Training of Final Model Complete! Save dir: eval-20190121-225901-REPRODUCTION_ATTEMPT_KEY_PADDING_2a88102_AUTOAUGMENT-cifar10-DARTS_PRIMITIVES_DIL_IS_SEPCONV


"""
DARTS_PRIMITIVES_DIL_IS_SEPCONV = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6), layout='cell')
SHARPSEPCONV_DARTS = DARTS_PRIMITIVES_DIL_IS_SEPCONV


"""
2019_02_20_03_15_15 epoch, 120, train_acc, 92.372000, valid_acc, 85.140000, train_loss, 0.221692, valid_loss, 0.451555, lr, 1.000000e-04, best_epoch, 106, best_valid_acc, 85.488000
Overview ***** best_epoch: 106 best_valid_acc: 85.49 ***** Progress: 100%|| 120/120 [32:46:14<00:00, 1050.83s/it]
2019_02_20_03_15_16 genotype =
2019_02_20_03_15_16 Search for Model Complete! Save dir: search-20190218-182855-MULTI_CHANNEL_SEARCH_2cdd546_search_weighting_max_w-cifar10-PRIMITIVES-OPS-0
"""

"""
export CUDA_VISIBLE_DEVICES="1" && python3 train_search.py --dataset cifar10 --batch_size 48 --save MULTI_CHANNEL_SEARCH_`git rev-parse --short HEAD`_search_weighting_generate_genotype --init_channels 36 --epochs 120 --cutout --autoaugment --seed 30 --weighting_algorithm max_w --multi_channel --load_genotype MULTI_CHANNEL_MAX_W
"""
MULTI_CHANNEL_MAX_W = Genotype(normal=[[[[0.031705040484666824, 0.03185523673892021], [0.03177893906831741, 0.03134448081254959], [0.03166716545820236, 0.031712375581264496], [0.029017647728323936, 0.03185588866472244]], [[0.029935121536254883, 0.03062211535871029], [0.030779147520661354, 0.03137855976819992], [0.029251324012875557, 0.0314946286380291], [0.03025045432150364, 0.03166542947292328]], [[0.03056972473859787, 0.03078850358724594], [0.03152700886130333, 0.03200467303395271], [0.03034617006778717, 0.030499886721372604], [0.03244936838746071, 0.03154132887721062]], [[0.03180636093020439, 0.03089098632335663], [0.031697362661361694, 0.03184359520673752], [0.03322812542319298, 0.03393062949180603], [0.030843427404761314, 0.029719246551394463]]], [[[0.03128419816493988, 0.031194495037198067], [0.03058856725692749, 0.03147657588124275], [0.031216900795698166, 0.03170192241668701], [0.031173493713140488, 0.03137543052434921]], [[0.031009048223495483, 0.031437475234270096], [0.03093050792813301, 0.032270822674036026], [0.0310247540473938, 0.031824786216020584], [0.03070482611656189, 0.031772714108228683]], [[0.03092092275619507, 0.03189568594098091], [0.031120117753744125, 0.03147495165467262], [0.03019261360168457, 0.03115837462246418], [0.030209895223379135, 0.032053761184215546]], [[0.03122881054878235, 0.032255493104457855], [0.03103666380047798, 0.03217817842960358], [0.03211415931582451, 0.03217983618378639], [0.031776174902915955, 0.027217810973525047]]], [[[0.031139764934778214, 0.031343117356300354], [0.031348712742328644, 0.0311704333871603], [0.031114375218749046, 0.031393397599458694], [0.03135799989104271, 0.031376857310533524]], [[0.031363870948553085, 0.03133983165025711], [0.03131990507245064, 0.031386278569698334], [0.03129395470023155, 0.03141247481107712], [0.03139813244342804, 0.03134961426258087]], [[0.03136737272143364, 0.03135362267494202], [0.031361229717731476, 0.03139583021402359], [0.03121591918170452, 0.03143315017223358], [0.03138110414147377, 0.031383417546749115]], [[0.03139226883649826, 0.03140714764595032], [0.030216185376048088, 0.031235458329319954], [0.0313195176422596, 0.03124529868364334], [0.031390056014060974, 0.02979360893368721]]], [[[0.031185977160930634, 0.0347476452589035], [0.031276751309633255, 0.03181716054677963], [0.03155045956373215, 0.03125309571623802], [0.03140901401638985, 0.030443238094449043]], [[0.03139765188097954, 0.031110990792512894], [0.031140387058258057, 0.0347706563770771], [0.031181395053863525, 0.031136374920606613], [0.03144082427024841, 0.030691733583807945]], [[0.031139248982071877, 0.030506059527397156], [0.031309712678194046, 0.030356379225850105], [0.031275711953639984, 0.03479333594441414], [0.031114540994167328, 0.028683168813586235]], [[0.031051797792315483, 0.029256442561745644], [0.03109460510313511, 0.02885911427438259], [0.03137160465121269, 0.026584701612591743], [0.03130620718002319, 0.0347440205514431]]]], normal_concat=[], reduce=[[[[0.03184084966778755, 0.03022356890141964], [0.031105801463127136, 0.031179415062069893], [0.030835246667265892, 0.03152412548661232], [0.030335750430822372, 0.03146739676594734]], [[0.0312722884118557, 0.030680980533361435], [0.03111124038696289, 0.03419611230492592], [0.030018579214811325, 0.0313275121152401], [0.030679989606142044, 0.030776049941778183]], [[0.03137199953198433, 0.03137969598174095], [0.03083980642259121, 0.03173601254820824], [0.0307187270373106, 0.03199375793337822], [0.03031729720532894, 0.03262593224644661]], [[0.03139664605259895, 0.03198783099651337], [0.03134854882955551, 0.032299160957336426], [0.031395602971315384, 0.033140938729047775], [0.02979094348847866, 0.02908225916326046]]], [[[0.03108000010251999, 0.03168174996972084], [0.031100235879421234, 0.03115084394812584], [0.0312507227063179, 0.031540852040052414], [0.03107316978275776, 0.031749702990055084]], [[0.03112880326807499, 0.03155761584639549], [0.030918046832084656, 0.031701233237981796], [0.030843490734696388, 0.03167719021439552], [0.030324121937155724, 0.0316300094127655]], [[0.031587373465299606, 0.03129081800580025], [0.031133320182561874, 0.03152334317564964], [0.03168809786438942, 0.03165533021092415], [0.03135747089982033, 0.03168513998389244]], [[0.031229818239808083, 0.03168436884880066], [0.031002137809991837, 0.031722329556941986], [0.03130635246634483, 0.031684760004282], [0.031463395804166794, 0.02757817879319191]]], [[[0.03080432116985321, 0.03146462142467499], [0.031355828046798706, 0.03139540180563927], [0.031218519434332848, 0.0312686488032341], [0.03146140277385712, 0.031395960599184036]], [[0.031333137303590775, 0.031254902482032776], [0.031280551105737686, 0.031448788940906525], [0.0313376784324646, 0.0313015952706337], [0.03138606250286102, 0.03128112107515335]], [[0.031144285574555397, 0.03141070157289505], [0.030991872772574425, 0.031274985522031784], [0.030961886048316956, 0.03148787468671799], [0.03144081309437752, 0.031142987310886383]], [[0.031290002167224884, 0.03111068159341812], [0.03080933541059494, 0.030690569430589676], [0.03141043335199356, 0.031086774542927742], [0.03126226365566254, 0.031495995819568634]]], [[[0.030837170779705048, 0.036189883947372437], [0.030746731907129288, 0.03082641214132309], [0.03060324676334858, 0.031165866181254387], [0.030544260516762733, 0.029803266748785973]], [[0.03094310127198696, 0.03066748008131981], [0.0308150053024292, 0.036842137575149536], [0.03100808709859848, 0.030431579798460007], [0.030625727027654648, 0.030208293348550797]], [[0.0308038592338562, 0.029830224812030792], [0.03025251068174839, 0.029800914227962494], [0.030794011428952217, 0.037474796175956726], [0.029639746993780136, 0.027296157553792]], [[0.03045455925166607, 0.02877660281956196], [0.030689792707562447, 0.028666401281952858], [0.030625801533460617, 0.027602000162005424], [0.030382489785552025, 0.04465189948678017]]]], reduce_concat=[], layout='raw_weights')

MULTI_CHANNEL_MAX_W_PATH = ['Source', 'Conv3x3_3', 'BatchNorm_3', 'layer_0_stride_1_c_in_256_c_out_128_op_type_ResizablePool', 'layer_0_add_c_out_128_stride_1', 'layer_0_stride_2_c_in_128_c_out_256_op_type_ResizablePool', 'layer_0_add_c_out_256_stride_2', 'layer_1_stride_1_c_in_256_c_out_32_op_type_ResizablePool', 'layer_1_add_c_out_32_stride_1', 'layer_1_stride_2_c_in_32_c_out_256_op_type_ResizablePool', 'layer_1_add_c_out_256_stride_2', 'layer_2_stride_1_c_in_256_c_out_256_op_type_SharpSepConv', 'layer_2_add_c_out_256_stride_1', 'layer_2_stride_2_c_in_256_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_2', 'layer_3_stride_1_c_in_256_c_out_256_op_type_ResizablePool', 'layer_3_add_c_out_256_stride_1', 'layer_3_stride_2_c_in_256_c_out_256_op_type_ResizablePool', 'layer_3_add_c_out_256_stride_2', 'SharpSepConv256', 'add-SharpSep', 'global_pooling', 'Linear']


"""
± export CUDA_VISIBLE_DEVICES="0" && python3 train_search.py --dataset cifar10 --batch_size 48 --save MULTI_CHANNEL_SEARCH_`git rev-parse --short HEAD`_search_weighting_original_darts --init_channels 36 --epochs 120 --cutout --autoaugment --seed 30 --weighting_algorithm scalar --multi_channel
2019_02_24_04_38_04 Search for Model Complete! Save dir: search-20190223-003007-MULTI_CHANNEL_SEARCH_938225a_search_weighting_original_darts-cifar10-PRIMITIVES-OPS-0
2019_02_24_03_56_17 epoch, 117, train_acc, 89.999998, valid_acc, 85.579998, train_loss, 0.294225, valid_loss, 0.424499, lr, 1.289815e-04, best_epoch, 117, best_valid_acc, 85.579998
2019_02_24_03_56_17 genotype =
"""
MULTI_CHANNEL_SCALAR = Genotype(normal=[[[[0.08428546786308289, 0.019649818539619446], [0.029455358162522316, 0.026891281828284264], [0.027873601764440536, 0.026621485128998756], [0.027566319331526756, 0.027208974584937096]], [[0.02464308589696884, 0.02305542305111885], [0.03325537592172623, 0.02161835879087448], [0.036233533173799515, 0.023530790582299232], [0.04525946453213692, 0.02315031923353672]], [[0.05040294677019119, 0.02056518942117691], [0.030842378735542297, 0.024993691593408585], [0.04064971208572388, 0.014905016869306564], [0.07537227869033813, 0.0182951632887125]], [[0.02582853101193905, 0.020147651433944702], [0.027859963476657867, 0.021051088348031044], [0.06657709181308746, 0.017026707530021667], [0.03375856950879097, 0.011425447650253773]]], [[[0.03005879931151867, 0.035663411021232605], [0.025182029232382774, 0.04607615992426872], [0.038091227412223816, 0.038178831338882446], [0.024393698200583458, 0.029284125193953514]], [[0.025640638545155525, 0.03322795405983925], [0.02253263257443905, 0.037514571100473404], [0.0183589868247509, 0.033491283655166626], [0.015858527272939682, 0.04169301316142082]], [[0.02627536468207836, 0.030852915719151497], [0.027742547914385796, 0.05268079787492752], [0.05241338908672333, 0.03677228465676308], [0.03717765957117081, 0.04306963086128235]], [[0.023739686235785484, 0.02377346344292164], [0.018741942942142487, 0.030636483803391457], [0.016585536301136017, 0.033416468650102615], [0.016086628660559654, 0.0347893163561821]]], [[[0.03236594796180725, 0.0732831209897995], [0.05554317310452461, 0.03167788311839104], [0.015518044121563435, 0.03799673169851303], [0.021326560527086258, 0.08206182718276978]], [[0.03787698224186897, 0.026206783950328827], [0.01189108844846487, 0.06706617027521133], [0.010926412418484688, 0.04731278121471405], [0.010900018736720085, 0.0639415979385376]], [[0.028068145737051964, 0.02306116558611393], [0.009883608669042587, 0.022720031440258026], [0.010248321108520031, 0.07221531122922897], [0.011321073397994041, 0.028630713000893593]], [[0.01708007976412773, 0.013192839920520782], [0.008508067578077316, 0.02285042405128479], [0.01724369265139103, 0.02031189575791359], [0.012589368969202042, 0.056180018931627274]]], [[[0.04276231303811073, 0.21309605240821838], [0.022647246718406677, 0.01787652261555195], [0.0067022559233009815, 0.010822849348187447], [0.0047399611212313175, 0.016015738248825073]], [[0.028339920565485954, 0.00891443807631731], [0.004901951644569635, 0.2276517152786255], [0.01379478070884943, 0.007287896703928709], [0.0033110049553215504, 0.011687851510941982]], [[0.018152590841054916, 0.006466378923505545], [0.022874118760228157, 0.011997316963970661], [0.013169433921575546, 0.10174134373664856], [0.006852362770587206, 0.009393713437020779]], [[0.005789881572127342, 0.0029638162814080715], [0.0041101728565990925, 0.0031539236661046743], [0.0014213536633178592, 0.002380270743742585], [0.0011982121504843235, 0.1477825939655304]]]], normal_concat=[], reduce=[[[[0.028326164931058884, 0.032454293221235275], [0.02686143107712269, 0.029518621042370796], [0.03313954547047615, 0.04891181364655495], [0.025960393249988556, 0.042642317712306976]], [[0.018554436042904854, 0.0273482296615839], [0.02059074118733406, 0.03055424988269806], [0.023065906018018723, 0.026168793439865112], [0.017394432798027992, 0.029353225603699684]], [[0.029137184843420982, 0.04715714231133461], [0.02788364700973034, 0.03222033008933067], [0.05128740146756172, 0.02894807606935501], [0.027765892446041107, 0.04117625951766968]], [[0.020344581454992294, 0.05894557386636734], [0.021750222891569138, 0.027226511389017105], [0.022402610629796982, 0.04984541982412338], [0.025405606254935265, 0.027658965438604355]]], [[[0.025873463600873947, 0.05752668157219887], [0.018333958461880684, 0.049362268298864365], [0.015626557171344757, 0.03343683108687401], [0.015159770846366882, 0.037002503871917725]], [[0.028882542625069618, 0.05437920615077019], [0.024060335010290146, 0.044413063675165176], [0.026867222040891647, 0.035240404307842255], [0.023217199370265007, 0.045194290578365326]], [[0.02042427659034729, 0.05703655630350113], [0.021850237622857094, 0.03791709989309311], [0.015018402598798275, 0.04467809945344925], [0.02511998824775219, 0.05425426736474037]], [[0.016492484137415886, 0.03154151141643524], [0.017121894285082817, 0.01935427449643612], [0.014856543391942978, 0.02986033819615841], [0.014744272455573082, 0.04515323415398598]]], [[[0.0597408190369606, 0.09758277982473373], [0.010528073646128178, 0.02998235821723938], [0.010792501270771027, 0.05588904768228531], [0.007236948702484369, 0.03156544268131256]], [[0.017808040603995323, 0.018275177106261253], [0.005358295980840921, 0.20611651241779327], [0.009535307064652443, 0.03866533190011978], [0.0031887770164757967, 0.01161785889416933]], [[0.006274309009313583, 0.01825445331633091], [0.013663525693118572, 0.016340306028723717], [0.007381833158433437, 0.17026008665561676], [0.0030986962374299765, 0.014190435409545898]], [[0.009256887249648571, 0.02237699367105961], [0.00712384469807148, 0.007060194853693247], [0.009489973075687885, 0.018868396058678627], [0.0026137656532227993, 0.05986303836107254]]], [[[0.0009627753752283752, 0.005614960100501776], [0.0009754917700774968, 0.0017817521002143621], [0.0009748890879563987, 0.0016846026992425323], [0.00192651420366019, 0.002251992467790842]], [[0.0012877885019406676, 0.0009805896552279592], [0.0016325361793860793, 0.052085988223552704], [0.0021779246162623167, 0.0018919521244242787], [0.000974901660811156, 0.0026733994018286467]], [[0.002875175792723894, 0.0009731581667438149], [0.0009763489360921085, 0.0015420051058754325], [0.0009765044087544084, 0.21534165740013123], [0.0009751239558681846, 0.002350056543946266]], [[0.015049988403916359, 0.000973944494035095], [0.002387009793892503, 0.0010801200987771153], [0.0009752536425366998, 0.0010066847316920757], [0.0009852066868916154, 0.671653687953949]]]], reduce_concat=[], layout='raw_weights')
MULTI_CHANNEL_SCALAR_PATH = ['Source', 'Conv3x3_0', 'BatchNorm_0', 'layer_0_stride_1_c_in_32_c_out_32_op_type_SharpSepConv', 'layer_0_add_c_out_32_stride_1','layer_0_stride_2_c_in_32_c_out_128_op_type_ResizablePool', 'layer_0_add_c_out_128_stride_2', 'layer_1_stride_1_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_1_add_c_out_128_stride_1', 'layer_1_stride_2_c_in_128_c_out_32_op_type_ResizablePool', 'layer_1_add_c_out_32_stride_2', 'layer_2_stride_1_c_in_32_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_1', 'layer_2_stride_2_c_in_256_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_2', 'layer_3_stride_1_c_in_256_c_out_256_op_type_ResizablePool', 'layer_3_add_c_out_256_stride_1', 'layer_3_stride_2_c_in_256_c_out_256_op_type_ResizablePool', 'layer_3_add_c_out_256_stride_2', 'SharpSepConv256', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_HANDMADE_PATH = ['Source', 'Conv3x3_0', 'BatchNorm_0', 'layer_0_stride_1_c_in_32_c_out_32_op_type_SharpSepConv', 'layer_0_add_c_out_32_stride_1', 'layer_0_stride_2_c_in_32_c_out_64_op_type_ResizablePool', 'layer_0_add_c_out_64_stride_2', 'layer_1_stride_1_c_in_64_c_out_64_op_type_SharpSepConv', 'layer_1_add_c_out_64_stride_1', 'layer_1_stride_2_c_in_64_c_out_128_op_type_ResizablePool', 'layer_1_add_c_out_128_stride_2', 'layer_2_stride_1_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_2_add_c_out_128_stride_1', 'layer_2_stride_2_c_in_128_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_2', 'layer_3_stride_1_c_in_256_c_out_256_op_type_SharpSepConv', 'layer_3_add_c_out_256_stride_1', 'layer_3_stride_2_c_in_256_c_out_256_op_type_ResizablePool', 'layer_3_add_c_out_256_stride_2', 'SharpSepConv256', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_GREEDY_SCALAR_TOP_DOWN = ['Source', 'Conv3x3_0', 'BatchNorm_0', 'layer_0_stride_1_c_in_32_c_out_64_op_type_ResizablePool', 'layer_0_add_c_out_64_stride_1', 'layer_0_stride_2_c_in_64_c_out_64_op_type_ResizablePool', 'layer_0_add_c_out_64_stride_2', 'layer_1_stride_1_c_in_64_c_out_256_op_type_ResizablePool', 'layer_1_add_c_out_256_stride_1', 'layer_1_stride_2_c_in_256_c_out_32_op_type_SharpSepConv', 'layer_1_add_c_out_32_stride_2', 'layer_2_stride_1_c_in_32_c_out_256_op_type_SharpSepConv', 'layer_2_add_c_out_256_stride_1', 'layer_2_stride_2_c_in_256_c_out_32_op_type_SharpSepConv', 'layer_2_add_c_out_32_stride_2', 'layer_3_stride_1_c_in_32_c_out_32_op_type_ResizablePool', 'layer_3_add_c_out_32_stride_1', 'layer_3_stride_2_c_in_32_c_out_64_op_type_ResizablePool', 'layer_3_add_c_out_64_stride_2', 'SharpSepConv64', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_GREEDY_SCALAR_BOTTOM_UP = ['Source', 'Conv3x3_2', 'BatchNorm_2', 'layer_0_stride_1_c_in_128_c_out_256_op_type_SharpSepConv', 'layer_0_add_c_out_256_stride_1', 'layer_0_stride_2_c_in_256_c_out_32_op_type_ResizablePool', 'layer_0_add_c_out_32_stride_2', 'layer_1_stride_1_c_in_32_c_out_32_op_type_ResizablePool', 'layer_1_add_c_out_32_stride_1', 'layer_1_stride_2_c_in_32_c_out_32_op_type_ResizablePool', 'layer_1_add_c_out_32_stride_2', 'layer_2_stride_1_c_in_32_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_1', 'layer_2_stride_2_c_in_256_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_2', 'layer_3_stride_1_c_in_256_c_out_256_op_type_ResizablePool', 'layer_3_add_c_out_256_stride_1', 'layer_3_stride_2_c_in_256_c_out_32_op_type_SharpSepConv', 'layer_3_add_c_out_32_stride_2', 'SharpSepConv32', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_GREEDY_MAX_W_TOP_DOWN = ['Source', 'Conv3x3_0', 'BatchNorm_0', 'layer_0_stride_1_c_in_32_c_out_32_op_type_ResizablePool', 'layer_0_add_c_out_32_stride_1', 'layer_0_stride_2_c_in_32_c_out_128_op_type_SharpSepConv', 'layer_0_add_c_out_128_stride_2', 'layer_1_stride_1_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_1_add_c_out_128_stride_1', 'layer_1_stride_2_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_1_add_c_out_128_stride_2', 'layer_2_stride_1_c_in_128_c_out_64_op_type_SharpSepConv', 'layer_2_add_c_out_64_stride_1', 'layer_2_stride_2_c_in_64_c_out_64_op_type_SharpSepConv', 'layer_2_add_c_out_64_stride_2', 'layer_3_stride_1_c_in_64_c_out_128_op_type_SharpSepConv', 'layer_3_add_c_out_128_stride_1', 'layer_3_stride_2_c_in_128_c_out_128_op_type_ResizablePool', 'layer_3_add_c_out_128_stride_2', 'SharpSepConv128', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_GREEDY_MAX_W_BOTTOM_UP = ['Source', 'Conv3x3_3', 'BatchNorm_3', 'layer_0_stride_1_c_in_256_c_out_128_op_type_ResizablePool', 'layer_0_add_c_out_128_stride_1', 'layer_0_stride_2_c_in_128_c_out_256_op_type_ResizablePool', 'layer_0_add_c_out_256_stride_2', 'layer_1_stride_1_c_in_256_c_out_128_op_type_ResizablePool', 'layer_1_add_c_out_128_stride_1', 'layer_1_stride_2_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_1_add_c_out_128_stride_2', 'layer_2_stride_1_c_in_128_c_out_64_op_type_ResizablePool', 'layer_2_add_c_out_64_stride_1', 'layer_2_stride_2_c_in_64_c_out_64_op_type_ResizablePool', 'layer_2_add_c_out_64_stride_2', 'layer_3_stride_1_c_in_64_c_out_64_op_type_ResizablePool', 'layer_3_add_c_out_64_stride_1', 'layer_3_stride_2_c_in_64_c_out_64_op_type_ResizablePool', 'layer_3_add_c_out_64_stride_2', 'SharpSepConv64', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_RANDOM_PATH = ['Source', 'Conv3x3_1', 'BatchNorm_1', 'layer_0_stride_1_c_in_64_c_out_128_op_type_ResizablePool', 'layer_0_add_c_out_128_stride_1', 'layer_0_stride_2_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_0_add_c_out_128_stride_2', 'layer_1_stride_1_c_in_128_c_out_256_op_type_ResizablePool', 'layer_1_add_c_out_256_stride_1', 'layer_1_stride_2_c_in_256_c_out_256_op_type_SharpSepConv', 'layer_1_add_c_out_256_stride_2', 'layer_2_stride_1_c_in_256_c_out_128_op_type_SharpSepConv', 'layer_2_add_c_out_128_stride_1', 'layer_2_stride_2_c_in_128_c_out_32_op_type_SharpSepConv', 'layer_2_add_c_out_32_stride_2', 'layer_3_stride_1_c_in_32_c_out_64_op_type_ResizablePool', 'layer_3_add_c_out_64_stride_1', 'layer_3_stride_2_c_in_64_c_out_32_op_type_SharpSepConv', 'layer_3_add_c_out_32_stride_2', 'SharpSepConv32', 'add-SharpSep', 'global_pooling', 'Linear']
MULTI_CHANNEL_RANDOM_OPTIMAL = ['Source', 'Conv3x3_1', 'BatchNorm_1', 'layer_0_stride_1_c_in_64_c_out_64_op_type_SharpSepConv', 'layer_0_add_c_out_64_stride_1', 'layer_0_stride_2_c_in_64_c_out_128_op_type_ResizablePool', 'layer_0_add_c_out_128_stride_2', 'layer_1_stride_1_c_in_128_c_out_32_op_type_ResizablePool', 'layer_1_add_c_out_32_stride_1', 'layer_1_stride_2_c_in_32_c_out_256_op_type_SharpSepConv', 'layer_1_add_c_out_256_stride_2', 'layer_2_stride_1_c_in_256_c_out_32_op_type_ResizablePool', 'layer_2_add_c_out_32_stride_1', 'layer_2_stride_2_c_in_32_c_out_256_op_type_ResizablePool', 'layer_2_add_c_out_256_stride_2', 'layer_3_stride_1_c_in_256_c_out_128_op_type_SharpSepConv', 'layer_3_add_c_out_128_stride_1', 'layer_3_stride_2_c_in_128_c_out_128_op_type_SharpSepConv', 'layer_3_add_c_out_128_stride_2', 'SharpSepConv128', 'add-SharpSep', 'global_pooling', 'Linear']

# costar@ubuntu|~/src/sharpDARTS/cnn on multi_channel_search!?
# ± export CUDA_VISIBLE_DEVICES="0" && python3 train_search.py --dataset cifar10 --batch_size 48 --layers_of_cells 8 --layers_in_cells 4 --save max_w_SharpSepConvDARTS_SEARCH_`git rev-parse --short HEAD` --init_channels
# 16 --epochs 120 --cutout --autoaugment --seed 22 --weighting_algorithm max_w --primitives DARTS_PRIMITIVES
# Tensorflow is not installed. Skipping tf related imports
# Experiment dir : search-20190321-024555-max_w_SharpSepConvDARTS_SEARCH_5e49783-cifar10-DARTS_PRIMITIVES-OPS-0
# 2019_03_21_19_11_44 epoch, 47, train_acc, 76.215997, valid_acc, 74.855997, train_loss, 0.687799, valid_loss, 0.734363, lr, 1.580315e-02, best_epoch, 47, best_valid_acc, 74.855997
# 2019_03_21_19_11_44 genotype = 
# 2019_03_21_19_11_44 alphas_normal = tensor([[0.1134, 0.0945, 0.0894, 0.1007, 0.1466, 0.1334, 0.1964, 0.1256],
#         [0.1214, 0.0983, 0.1000, 0.1072, 0.1675, 0.1383, 0.1167, 0.1507],
#         [0.1251, 0.1066, 0.1043, 0.1674, 0.1295, 0.1237, 0.1209, 0.1225],
#         [0.1238, 0.1066, 0.1054, 0.1108, 0.1331, 0.1418, 0.1145, 0.1641],
#         [0.1009, 0.0843, 0.0801, 0.0802, 0.2970, 0.1168, 0.1329, 0.1078],
#         [0.1257, 0.1115, 0.1087, 0.1158, 0.1641, 0.1312, 0.1305, 0.1125],
#         [0.1662, 0.1154, 0.1152, 0.1194, 0.1234, 0.1248, 0.1222, 0.1134],
#         [0.1177, 0.0943, 0.1892, 0.0836, 0.1285, 0.1317, 0.1286, 0.1263],
#         [0.3851, 0.0835, 0.0718, 0.0554, 0.1031, 0.1047, 0.1011, 0.0953],
#         [0.1249, 0.1119, 0.1096, 0.1156, 0.1284, 0.1177, 0.1157, 0.1762],
#         [0.1249, 0.1186, 0.1197, 0.1244, 0.1254, 0.1319, 0.1238, 0.1312],
#         [0.1126, 0.0932, 0.2448, 0.0838, 0.1132, 0.1141, 0.1263, 0.1120],
#         [0.0791, 0.4590, 0.0665, 0.0518, 0.0843, 0.0804, 0.0846, 0.0944],
#         [0.0715, 0.0665, 0.4912, 0.0401, 0.0822, 0.0816, 0.0888, 0.0780]],
#        device='cuda:0', grad_fn=<SoftmaxBackward>)
SHARPSEPCONV_DARTS_MAX_W = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6), layout='cell')
