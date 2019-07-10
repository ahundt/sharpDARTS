import argparse
import os
import datetime

from model import CMC, TDC, TDCFeaturizer, CMCFeaturizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

try:
    from tensorboardX import SummaryWriter
    from tensorboardX import embedding
    from tensorboardX.x2num import make_np
except ImportError:
    ImportError('Install tensorboardX-1.7 by using "pip install tensorboardX" or' 
                ' see https://github.com/lanpa/tensorboardX for details')
try:
    from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset  
except ImportError:
    ImportError('The costar dataset is not available. '
                'See https://github.com/ahundt/costar_dataset for details')

def save_features_to_visualize(embeddings, experiment_name='default'):
    """
    Save the embeddings to be visualised using t-sne on TensorBoardX
    Reference: https://medium.com/@vegi/visualizing-higher-dimensional-data-using-t-sne-on-tensorboard-7dbf22682cf2
    Args: embeddings - of shape - (length of video, length of feature_vector)   
                                                    default length of feature_vector = 1024 for TDC and 2048 for CMC
    """
    vid_embed = Variable(torch.tensor(np.squeeze(np.concatenate(embeddings, 0))))

    # Generate metadata
    metadata = 'video_index\tframe_index\n'
    for video_index, video_embedding in enumerate(embeddings):
        print(video_index,' video_embedding length: ', len(video_embedding))
        for frame_index, frame_embedding in enumerate(video_embedding):
            metadata += '{}\t{}\n'.format(video_index, frame_index)
    
    currentDT = datetime.datetime.now()
    path = 'runs/' + str(currentDT)[:10] + '-' + str(currentDT)[11:16]
    subdir = experiment_name
    save_path = os.path.join(path,subdir)
    try:
        os.makedirs(save_path)
    except OSError:
        print('warning: Embedding dir exists, did you set global_step for add_embedding()?')
   
    metadata_path = os.path.join(save_path, 'metadata.tsv')
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(metadata)
    
    writer = SummaryWriter(write_to_disk=False)
    mat = make_np(vid_embed)
    embedding.make_mat(mat,save_path)
    embedding.append_pbtxt(metadata,label_img=None,save_path=path,subdir=subdir,global_step=0,tag='default')
    writer.close()
    print("Saved embeddings in: ",save_path)
    return

def generate_features(videos_path, model_path, batch_size,feature_mode, version, set_name, subset_name):
    """
    Generates and saves the embeddings for all the videos listed in the file -videos_path. 
    Args: videos_path - path to the file where the videos are listed
          model_path - path to the best trained model, assumes it is named model_best.pth.tar
          batch_size - batch size per process. Defaults to 32
          feature_mode - "cross_modal_embeddings" or "time_difference_images". Default is 'cross_modal_embeddings'
          version - the CoSTAR BSD version to use. Defaults to "v0.4"
          set_name - which set to use in the CoSTAR BSD. Options are "blocks_only" or "blocks_with_plush_toy".
                    Defaults to "blocks_only"
          subset_name - which subset to use in the CoSTAR BSD. Options are "success_only",
                        "error_failure_only", "task_failure_only", or "task_and_error_failure". Defaults to "success_only"
    """
    datasets = CostarBlockStackingDataset.from_standard_txt(root=videos_path,
                version=version, set_name=set_name, subset_name=subset_name,
                split='test', feature_mode=feature_mode, output_shape=(3, 96, 128),
                num_images_per_example=200, is_training=False,visual_mode=True)
    
    loaders = [DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1) for dataset in datasets]
    print("Length of the dataset: {}. Length of the loader: {}.".format(len(datasets), len(loaders)))
    
    best_model_path = os.path.join(args.model_path, 'model_best.pth.tar')
    pretrained_model = torch.load(best_model_path,map_location='cpu')
       
    model_tdc = TDCFeaturizer().double()
    model_tdc_dict = model_tdc.state_dict()
    pretrained_tdc_dict = {k: v for k, v in pretrained_model.items() if k in model_tdc_dict}
    model_tdc_dict.update(pretrained_tdc_dict)
    model_tdc.load_state_dict(model_tdc_dict)

    if(feature_mode == 'cross_modal_embeddings'):
        model_cmc = CMCFeaturizer().double()
        model_cmc_dict = model_cmc.state_dict()
        pretrained_cmc_dict = {k: v for k, v in pretrained_model.items() if k in model_cmc_dict}
        model_cmc_dict.update(pretrained_cmc_dict)
        model_cmc.load_state_dict(model_cmc_dict)

    features_all = []    
    for loader in loaders:
        feature_vectors = []
        for _,batch in enumerate(loader):
            if (feature_mode == 'cross_modal_embeddings'):
                frame,joint = batch
                frame_features = model_tdc(frame)
                joint_features = model_cmc(joint)	
                features = torch.cat((frame_features, joint_features),1)
            else: 
                frame = batch
                features = model_tdc(frame)
            features = F.normalize(features).cpu().detach().numpy()
            feature_vectors.append(features)
        feature_vecs = np.concatenate(feature_vectors)
        features_all.append(feature_vecs)   
    return features_all

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data', help='Path to dataset', default='~/.keras/datasets/costar_block_stacking_dataset_v0.4')
    args_parser.add_argument('--model_path', help='Path for best model',default='')
    args_parser.add_argument('--batch_size', help='Batch size for visualizing', type=int, default=32)
    args_parser.add_argument('--feature_mode', help = 'cross_modal_embeddings or time_difference_images',default='cross_modal_embeddings')
    args_parser.add_argument('--version', help='Path for best model',default='v0.4')
    args_parser.add_argument('--set_name', help='Path for best model',default='blocks_only')
    args_parser.add_argument('--subset_name', help='Path for best model',default='success_only')
    args = args_parser.parse_args()
    
    features_all = generate_features(args.data, args.model_path, args.batch_size, args.feature_mode,
                   args.version, args.set_name, args.subset_name)
    save_features_to_visualize(features_all)    