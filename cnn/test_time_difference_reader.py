import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm
try:
    from costar_dataset.block_stacking_reader_torch import CostarBlockStackingDataset
except ImportError:
    ImportError('The costar dataset is not available. '
                'See https://github.com/ahundt/costar_dataset for details')

if __name__ == '__main__':
    """ To test block_stacking_reader for feature_modes time_difference_images and cross_modal_embeddings"""
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--videos_path', help='Path for training data')
    args_parser.add_argument('--feature_mode', help = 'cross_modal_embeddings or time_difference_images',default='time_difference_images')
    args_parser.add_argument('--batch_size', help='Batch size for training', type=int, default=32)
    args_parser.add_argument('--visualize', help='To view frames, set true', default=True)
    args = args_parser.parse_args()

    visualize = args.visualize

    costar_dataset = CostarBlockStackingDataset.from_standard_txt(
                      root=args.videos_path,
                      version='v0.4', set_name='blocks_only', subset_name='success_only',
                      split='train', feature_mode=args.feature_mode, output_shape=(3, 96, 128),
                      num_images_per_example=200, is_training=False)
    generator = DataLoader(costar_dataset, args.batch_size, shuffle=False, num_workers=4)
    print("Length of the dataset: {}. Length of the loader: {}.".format(len(costar_dataset), len(generator)))

    generator_output = iter(generator)
    print("-------------------op")
    x1, x2, y = next(generator_output)
    print("Image 1 shape: ", x1.shape, "  Image 2/Joint shape: ",x2.shape, "  Labels shape: ", y.shape)
    
    pb = tqdm(range(len(generator)-1))
    for i in pb:
        pb.set_description('batch: {}'.format(i))

        x1, x2, y = generator_output.next()
        y = y.numpy()
        x1 = [t.numpy() for t in x1]
        x2 = [t.numpy() for t in x2]
        distances = ['0', '1', '2','3 or 4', 'btw 5 and 20', 'btw 21 and 150']
        if visualize:
            import matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure()
            if (args.feature_mode == 'time_difference_images'):
                title = "Interval between frames is " + str(distances[y[0]])
            else:
                title = "Interval between frame and joint_vector is " + str(distances[y[0]])
            plt.title(title)            
            img1 = np.moveaxis(x1[0], 0, 2)
            img2 = np.moveaxis(x2[0], 0, 2)
            
            # image 1
            fig1 = fig.add_subplot(1,2,1)
            fig1.set_title("Frame 1")
            plt.imshow(img1)
            plt.draw()
            plt.pause(0.25)

            if (args.feature_mode == 'time_difference_images'):
                # image 2
                fig2 = fig.add_subplot(1,2,2)
                fig2.set_title("Frame 2")
                plt.imshow(img2)
                plt.draw()
                plt.pause(0.25)
            # uncomment the following line to wait for one window to be closed before showing the next    
            plt.show()

        assert np.all(x1[0] <= 1) and np.all(x1[0] >= -1), "x1[0] is not within range!"
        assert np.all(x1[1] <= 1) and np.all(x1[1] >= -1), "x1[1] is not within range!"
        assert np.all(x1[2] <= 1) and np.all(x1[2] >= 0), "x1[2] is not within range!"
        assert np.all(y <= 5) and np.all(y >= 0), "y is not within range!"
