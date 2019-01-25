# Set matplotlib backend to Agg
# *MUST* be done BEFORE importing hiddenlayer or libs that import matplotlib
import matplotlib
matplotlib.use("Agg")

import os
import torch
# import networkx
import model_search
# requires https://github.com/waleedka/hiddenlayer
import hiddenlayer as hl

print('initializing module')
cnn_model = model_search.MultiChannelNetwork(always_apply_ops=True, layers=3, steps=3, visualization=True)
transforms = [
  hl.transforms.Fold('MaxPool3x3 > Conv1x1 > BatchNorm', 'ResizableMaxPool', 'ResizableMaxPool'),
  hl.transforms.Fold('MaxPool > Conv > BatchNorm', 'ResizableMaxPool', 'ResizableMaxPool'),
  hl.transforms.Fold('Relu > Conv > Conv > BatchNorm', 'ReluSepConvBn'),
  hl.transforms.Fold('ReluSepConvBn > ReluSepConvBn', 'SharpSepConv', 'SharpSepConv'),
  hl.transforms.Prune('Constant'),
  hl.transforms.Prune('Gather'),
  hl.transforms.Prune('Unsqueeze'),
  hl.transforms.Prune('Concat'),
  hl.transforms.Prune('Shape'),
# Fold repeated blocks
  hl.transforms.FoldDuplicates(),
]
print('building graph')
# WARNING: the code may hang here. These are instructions for a workaround:
# First install hiddenlayer from source:
#
#     cd ~/src
#     git clone https://github.com/waleedka/hiddenlayer.git
#     cd hiddenlayer
#     pip3 install --user --upgrade -e .
#
# Next open the file /hiddenlayer/hiddenlayer/pytorch_builder.py
#
# change:
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
# to
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.RAW)
#
# The graph is very large so building the graph will take a long time.
# Note that at the time of writing the graph algorithms can't handle multiplying by a constant.
# Instead, I added if statements that skip the weight component if it is in visualization mode.
#
# For progress bars go back to /hiddenlayer/hiddenlayer/pytorch_builder.py:
# at the top add:
#     import tqdm as tqdm
#
# Then in:
#
#     def import_graph()
#
# find all instances of:
#
#    torch_graph.nodes()
#
# and replace with:
#
#    tqdm(torch_graph.nodes())
#
cnn_graph = hl.build_graph(cnn_model, torch.zeros([2, 3, 32, 32]), transforms=transforms)
output_file = os.path.expanduser('~/src/darts/cnn/multi_channel_network.pdf')
print('build complete, saving: ' + output_file)
cnn_graph.save(output_file)
print('save complete')