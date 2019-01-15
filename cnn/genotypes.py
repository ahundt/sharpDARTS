from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Primitives for the dilation, sep_conv, flood, and choke 3x3 only search space
# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     # 'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     # 'sep_conv_5x5',
#     'dil_conv_3x3',
#     # 'dil_conv_5x5',
#     # 'nor_conv_3x3',
#     # 'nor_conv_5x5',
#     # 'nor_conv_7x7',
#     'flood_conv_3x3',
#     'dil_flood_conv_3x3',
#     'choke_conv_3x3',
#     'dil_choke_conv_3x3',
# ]

# Primitives for the original darts search space
# DARTS_PRIMITIVES = [
PRIMITIVES = [
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
"""
DARTS_PRIMITIVES_DIL_IS_SEPCONV = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
