from dataset import *

dataset_name = 'stacking'
train_transform = None
dataset_location = '~/.keras/datasets/costar_block_stacking_dataset_v0.4'
batch_size = 3
train_proportion = 0.5
search_architecture = False
costar_version = 'v0.4'
costar_set_name = 'blocks_only'
costar_subset_name = 'success_only'
costar_feature_mode = 'translation_only'
costar_output_shape = (224, 224, 3)
costar_random_augmentation = None
costar_one_hot_encoding = True

train_queue, valid_queue = get_training_queues(
                                dataset_name=dataset_name,
                                train_transform=train_transform,
                                dataset_location=dataset_location,
                                batch_size=batch_size,
                                train_proportion=train_proportion,
                                search_architecture=search_architecture, 
                                costar_version=costar_version,
                                costar_set_name=costar_set_name,
                                costar_subset_name=costar_subset_name,
                                costar_feature_mode=costar_feature_mode,
                                costar_output_shape=costar_output_shape,
                                costar_random_augmentation=costar_random_augmentation,
                                costar_one_hot_encoding=costar_one_hot_encoding)
test_queue = get_costar_test_queue(
        dataset_location=dataset_location,
        batch_size=batch_size,
        costar_version=costar_version,
        costar_set_name=costar_set_name,
        costar_subset_name=costar_subset_name,
        costar_feature_mode=costar_feature_mode,
        costar_output_shape=costar_output_shape,
        costar_random_augmentation=costar_random_augmentation,
        costar_one_hot_encoding=costar_one_hot_encoding,
        verbose=1)

print("TRAIN: ")
for output in train_queue:
    print("-------------------op")
    x, y = output

    print(x.shape)
    print(y.shape)

    print("-------------------")

print("VALID: ")
for output in valid_queue:
    print("-------------------op")
    x, y = output

    print(x.shape)
    print(y.shape)

    print("-------------------")

print("TEST: ")
for output in test_queue:
    print("-------------------op")
    x, y = output

    print(x.shape)
    print(y.shape)

    print("-------------------")