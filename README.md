# sharpDARTS: Faster, More Accurate Differentiable Architecture Search

Please cite sharpDARTS if you use this code as part of your research! By using any part of the code you are agreeing to comply with our permissive Apache 2.0 license.

```
@article{DBLP:journals/corr/abs-1903-09900,
  author    = {Andrew Hundt and
               Varun Jain and
               Gregory D. Hager},
  title     = {sharpDARTS: Faster and More Accurate Differentiable Architecture Search},
  journal   = {CoRR},
  volume    = {abs/1903.09900},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.09900},
  archivePrefix = {arXiv},
  eprint    = {1903.09900},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-09900},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

For algorithm details see:
[sharpDARTS: Faster, More Accurate Differentiable Architecture Search](http://arxiv.org/abs/1903.09900)

Requires pytorch 1.0.


## Searching for a model

To see the configuration options, run `python3 train_search.py --help` from the directory `cnn`. The code is also commented.

Run `cnn/train_search.py` with your configuration. Place the best genotype into `genotypes.py`, preferably with a record of the raw weights as well and other command execution data so similar results can be reproduced in the future.

## Training a final Model

To see the configuration options, run `python3 train.py --help` from the directory `cnn`. The code is also commented.

### CIFAR-10 Training

Best Results with SharpSepConvDARTS:
```
export CUDA_VISIBLE_DEVICES="0" && python3 train.py --autoaugment --auxiliary --cutout --batch_size 64 --epochs 2000 --save SHARPSEPCONV_DARTS_2k_`git rev-parse --short HEAD`_cospower_min_1e-8 --arch SHARPSEPCONV_DARTS --learning_rate 0.025 --learning_rate_min 1e-8 --cutout_length 16 --init_channels 36 --dataset cifar10
```

Collecting multiple data points:

```
for i in {1..8}; do export CUDA_VISIBLE_DEVICES="0" && python3 train.py --autoaugment --auxiliary --cutout --batch_size 64 --epochs 2000 --save SHARPSEPCONV_DARTS_MAX_W_2k_`git rev-parse --short HEAD`_cospower_min_1e-8 --learning_rate 0.025 --learning_rate_min 1e-8 --cutout_length 16 --init_channels 36 --dataset cifar10 --arch SHARPSEPCONV_DARTS_MAX_W ; done;
```

Collecting results from multiple runs into a file from `sharperDARTS/cnn` directory with [ack-grep](https://beyondgrep.com/) (on some machiens it is just `ack`):
```
ack-grep  --match "cifar10.1" */*.txt > cifar10.1_results_femur.txt
```

### ImageNet Training

We train with a modified version the fp16 optimizer from [APEX](https://github.com/NVIDIA/apex), we suspect this means results will be slightly below the ideal possible accuracy, but training time is substantially reduced.
These configurations are designed for an NVIDIA RTX 1080Ti, `--fp16` should be removed for older GPUs.

You'll need to first [acquire the imagenet files](http://image-net.org/download) and preprocess them.

Best SharpSepConvDARTS Results:

```
python3 -m torch.distributed.launch --nproc_per_node=2 main_fp16_optimizer.py --fp16 --b 224 --save `git rev-parse --short HEAD`_DARTS_PRIMITIVES_DIL_IS_SEPCONV --epochs 300 --dynamic-loss-scale --workers 20 --autoaugment --auxiliary --cutout --data /home/costar/datasets/imagenet/ --learning_rate 0.05 --learning_rate_min 0.00075 --arch DARTS_PRIMITIVES_DIL_IS_SEPCONV
```

Resuming Training (you may need to do this if your results don't match the paper on the first run):

You'll need to make sure to specify the correct checkpoint directory, which will change every run.
```
python3 -m torch.distributed.launch --nproc_per_node=2 main_fp16_optimizer.py --fp16 --b 224 --save `git rev-parse --short HEAD`_DARTS_PRIMITIVES_DIL_IS_SEPCONV --epochs 100 --dynamic-loss-scale --workers 20 --autoaugment --auxiliary --cutout --data /home/costar/datasets/imagenet/ --learning_rate 0.0005 --learning_rate_min 0.0000075 --arch DARTS_PRIMITIVES_DIL_IS_SEPCONV --resume eval-20190213-182625-975c657_DARTS_PRIMITIVES_DIL_IS_SEPCONV-imagenet-DARTS_PRIMITIVES_DIL_IS_SEPCONV-0/checkpoint.pth.tar
```

Training SHARP_DARTS model on ImageNet:

```
export CUDA_VISIBLE_DEVICES="0" && python3 main_fp16_optimizer.py --fp16 --b 256 --save `git rev-parse --short HEAD` --epochs 400 --dynamic-loss-scale --workers 20 --autoaugment --auxiliary --cutout --data /home/costar/datasets/imagenet/ --learning_rate_min 1e-8 --mid_channels 96 --lr 1e-4 --load eval-20190222-175718-4f5892f-imagenet-SHARP_DARTS-0/model_best.pth.tar
```

## Differentiable Hyperparameter Search on CIFAR-10

Searching for a Model:
```
export CUDA_VISIBLE_DEVICES="1" && python2 train_search.py --dataset cifar10 --batch_size 64 --save MULTI_CHANNEL_SEARCH_`git rev-parse --short HEAD`_search_weighting_max_w --init_channels 36 --epochs 120 --cutout --autoaugment --seed 10 --weighting_algorithm max_w --multi_channel
```

Add the best path printed during the search process to `genotypes.py`, you can see examples in that file.

Training a final model, this one will reproduce the best results from Max-W search:
```
for i in {1..5}; do export CUDA_VISIBLE_DEVICES="0" && python3 train.py --b 512 --save MULTI_CHANNEL_MAX_W_PATH_FULL_2k_`git rev-parse --short HEAD`_LR_0.1_to_1e-8 --arch MULTI_CHANNEL_MAX_W_PATH --epochs 2000 --multi_channel --cutout --autoaugment --learning_rate 0.1 ; done;
```

# sharpDARTS Repository based on Differentiable Architecture Search (DARTS)

Code accompanying the paper
> [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)\
> Hanxiao Liu, Karen Simonyan, Yiming Yang.\
> _arXiv:1806.09055_.

<p align="center">
  <img src="img/darts.png" alt="darts" width="48%">
</p>
The algorithm is based on continuous relaxation and gradient descent in the architecture space. It is able to efficiently design high-performance convolutional architectures for image classification (on CIFAR-10 and ImageNet) and recurrent architectures for language modeling (on Penn Treebank and WikiText-2). Only a single GPU is required.

## Requirements
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```
NOTE: PyTorch 0.4 is not supported at this moment and would lead to OOM.

## Datasets
Instructions for acquiring PTB and WT2 can be found [here](https://github.com/salesforce/awd-lstm-lm). While CIFAR-10 can be automatically downloaded by torchvision, ImageNet needs to be manually downloaded (preferably to a SSD) following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Pretrained models
The easist way to get started is to evaluate our pretrained DARTS models.

**CIFAR-10** ([cifar10_model.pt](https://drive.google.com/file/d/1Y13i4zKGKgjtWBdC0HWLavjO7wvEiGOc/view?usp=sharing))
```
cd cnn && python test.py --auxiliary --model_path cifar10_model.pt
```
* Expected result: 2.63% test error rate with 3.3M model params.

**PTB** ([ptb_model.pt](https://drive.google.com/file/d/1Mt_o6fZOlG-VDF3Q5ModgnAJ9W6f_av2/view?usp=sharing))
```
cd rnn && python test.py --model_path ptb_model.pt
```
* Expected result: 55.68 test perplexity with 23M model params.

**ImageNet** ([imagenet_model.pt](https://drive.google.com/file/d/1AKr6Y_PoYj7j0Upggyzc26W0RVdg4CVX/view?usp=sharing))
```
cd cnn && python test_imagenet.py --auxiliary --model_path imagenet_model.pt
```
* Expected result: 26.7% top-1 error and 8.7% top-5 error with 4.7M model params.

## Architecture search (using small proxy models)
To carry out architecture search using 2nd-order approximation, run
```
cd cnn && python train_search.py --unrolled     # for conv cells on CIFAR-10
cd rnn && python train_search.py --unrolled     # for recurrent cells on PTB
```
Note the _validation performance in this step does not indicate the final performance of the architecture_. One must train the obtained genotype/architecture from scratch using full-sized models, as described in the next section.

Also be aware that different runs would end up with different local minimum. To get the best result, it is crucial to repeat the search process with different seeds and select the best cell(s) based on validation performance (obtained by training the derived cell from scratch for a small number of epochs). Please refer to fig. 3 and sect. 3.2 in our arXiv paper.

<p align="center">
<img src="img/progress_convolutional_normal.gif" alt="progress_convolutional_normal" width="29%">
<img src="img/progress_convolutional_reduce.gif" alt="progress_convolutional_reduce" width="35%">
<img src="img/progress_recurrent.gif" alt="progress_recurrent" width="33%">
</p>
<p align="center">
Figure: Snapshots of the most likely normal conv, reduction conv, and recurrent cells over time.
</p>

## Architecture evaluation (using full-sized models)
To evaluate our best cells by training from scratch, run
```
cd cnn && python train.py --auxiliary --cutout            # CIFAR-10
cd rnn && python train.py                                 # PTB
cd rnn && python train.py --data ../data/wikitext-2 \     # WT2
            --dropouth 0.15 --emsize 700 --nhidlast 700 --nhid 700 --wdecay 5e-7
cd cnn && python train_imagenet.py --auxiliary            # ImageNet
```
Customized architectures are supported through the `--arch` flag once specified in `genotypes.py`.

The CIFAR-10 result at the end of training is subject to variance due to the non-determinism of cuDNN back-prop kernels. _It would be misleading to report the result of only a single run_. By training our best cell from scratch, one should expect the average test error of 10 independent runs to fall in the range of 2.76 +/- 0.09% with high probability.

<p align="center">
<img src="img/cifar10.png" alt="cifar10" width="36%">
<img src="img/imagenet.png" alt="ptb" width="29%">
<img src="img/ptb.png" alt="ptb" width="30%">
</p>
<p align="center">
Figure: Expected learning curves on CIFAR-10 (4 runs), ImageNet and PTB.
</p>

## Visualization
Package [graphviz](https://graphviz.readthedocs.io/en/stable/index.html) is required to visualize the learned cells
```
python visualize.py DARTS
```
where `DARTS` can be replaced by any customized architectures in `genotypes.py`.

## Citation
If you use any part of this code in your research, please cite our [paper](https://arxiv.org/abs/1806.09055):
```
@article{liu2018darts,
  title={DARTS: Differentiable Architecture Search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  journal={arXiv preprint arXiv:1806.09055},
  year={2018}
}
```
