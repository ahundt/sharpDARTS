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

2019_01_21_23_21_59 gpu device = 0
2019_01_21_23_21_59 args = Namespace(arch='DARTS', autoaugment=True, auxiliary=True, auxiliary_weight=0.4, batch_size=64, cutout=True, cutout_length=16, data='../data', dataset='cifar10', drop_path
_prob=0.2, epochs=1000, gpu=0, grad_clip=5, init_channels=36, layers=20, learning_rate=0.025, learning_rate_min=1e-07, mixed_auxiliary=False, model_path='saved_models', momentum=0.9, ops='OPS', opt
imizer='sgd', partial=0.125, primitives='PRIMITIVES', random_eraser=False, report_freq=50, save='eval-20190121-232159-AUTOAUGMENT_V2_KEY_PADDING_d5dda02_BUGFIX-cifar10-DARTS', seed=4, warm_restarts
=20, weight_decay=0.0003)
2019_01_21_23_22_02 param size = 3.529270MB
2019_01_25_20_26_22 best_epoch, 988, best_train_acc, 95.852000, best_valid_acc, 97.890000, best_train_loss, 0.196667, best_valid_loss, 0.076396, lr, 8.881592e-06, best_epoch, 988, best_valid_acc, 97.890000 cifar10.1_valid_acc, 93.750000, cifar10.1_valid_loss, 0.218554
2019_01_25_20_26_22 Training of Final Model Complete! Save dir: eval-20190121-232159-AUTOAUGMENT_V2_KEY_PADDING_d5dda02_BUGFIX-cifar10-DARTS
'''
CHOKE_FLOOD_DIL_IS_SEP_CONV = Genotype(normal=[('choke_conv_3x3', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('choke_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('flood_conv_3x3', 0)], reduce_concat=range(2, 6))
SHARP_DARTS = CHOKE_FLOOD_DIL_IS_SEP_CONV

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

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
DARTS_PRIMITIVES_DIL_IS_SEPCONV = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
SHARPSEPCONV_DARTS = DARTS_PRIMITIVES_DIL_IS_SEPCONV