import numpy as np
import os
import glob
import argparse

ANGLE_ERROR_BINS = [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 240.0, 360.0]  # degrees
ANGLE_ERROR_HEADERS = '5deg, 15deg, 30deg, 60deg, 120deg, 240deg, 360deg'

CART_ERROR_BINS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12]  # meters
CART_ERROR_HEADERS = '5mm, 1cm, 2cm, 4cm, 8cm, 16cm, 32cm, 64cm, 128cm, 256cm, 512cm'


def _parse_args():
    parser = argparse.ArgumentParser(
            description='Summarize errors into bins and report the percentage of data in each bin.')
    parser.add_argument('-p', '--path', type=str, metavar='DIR',
                        help='Path to the save folder of a run from train_costar.py')
    return parser.parse_args()


def sort_data_into_bins(csv_filename):
    print('Processing {}'.format(csv_filename))

    # Determine the error type
    if 'cart' in csv_filename:
        error_bins = CART_ERROR_BINS
        header = CART_ERROR_HEADERS
    elif 'angle' in csv_filename:
        error_bins = ANGLE_ERROR_BINS
        header = ANGLE_ERROR_HEADERS
    else:
        raise ValueError('Unable to determine error type because filename '
                         'does not have either "cart" or "angle": {}'.format(csv_filename))

    # Load the csv file into numpy array
    epoch_error = np.genfromtxt(csv_filename, delimiter=',')
    if epoch_error.ndim == 1:
        # There is only one epoch -- possibly as a result of evaluation
        # In this case, unsqueeze the first dimension.
        # For example, from (49300,) to (1, 49300)
        epoch_error = epoch_error[np.newaxis, :]
    num_epochs, num_examples = epoch_error.shape
    print('- Read {} epochs, each epoch have {} examples'.format(num_epochs, num_examples))

    # Convert from radians to degrees for angle errors
    if 'angle' in csv_filename:
        epoch_error = epoch_error * (180.0 / np.pi)

    bin_result = np.zeros((num_epochs, len(error_bins)-1))
    for epoch in range(num_epochs):
        # Use np.histogram to return the count of elements in each bin
        hist, _ = np.histogram(epoch_error[epoch], error_bins)
        bin_result[epoch] = hist / num_examples

    # Save the result
    # Extract the filename from full path and add "summary_" to the front
    # For example, "summary_train_abs_angle_error.csv"
    path, csv_filename = os.path.split(csv_filename)
    csv_filename = os.path.join(path, 'summary_' + csv_filename)  

    np.savetxt(csv_filename, bin_result, fmt='%1.5f', delimiter=', ', header=header)
    print("- Summary file saved as {}".format(csv_filename))

    return bin_result


def main(args):
    args.path = os.path.expanduser(args.path)  # Expand the path
    if not os.path.isdir(args.path):
        raise ValueError('Input path is not a directory: {}'.format(args.path))

    csv_files = glob.glob(os.path.join(args.path, '*error.csv'))  # Get all csv files in this folder
    csv_files = [f for f in csv_files if 'summary' not in f]  # Skip existing summary files
    if not csv_files:
        raise RuntimeError('No error .csv file found in {}'.format(args.path))

    print("Counted {} error .csv files".format(len(csv_files)))

    for csv_file in csv_files:
        sort_data_into_bins(csv_file)


if __name__ == '__main__':
    main(_parse_args())
