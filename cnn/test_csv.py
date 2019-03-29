import os
import sys
import json
import csv
import utils
import numpy as np

test_file = 'cnn/eval-epoch-stats-20190204-114511.json'
if len(sys.argv) > 1:
    for s in sys.argv:
        if '.json' in s:
            test_file = s

print('loading ' + test_file)
output_file = test_file.replace('.json', '.csv')
if not os.path.exists(test_file):
    raise ValueError('File does not exist! ' + test_file)

with open(test_file, 'r') as f:
    epoch_stats = json.load(f)

utils.list_of_dicts_to_csv(output_file, epoch_stats)

print('saved file: ' + str(output_file))
