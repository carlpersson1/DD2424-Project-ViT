import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Reshape, Input, Dense
import numpy as np
import pickle
from typing import Dict, Union, Any


def get_cifar_dataset_scaled(
        key: str = "data_batch",
        testkey: str = "test_batch"
) -> Dict[str, Union[Union[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Any]]:

        def unpickle(file: 'File'):
            with open(file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            return data

        # Load training data
        training_data_batches = []
        train_data_images_batches = []
        train_data_labels_batches = []

        for i in range(1, 6):
            training_data_batch = unpickle(file=f'Datasets/{key}_{i}')
            training_data_batches.append(training_data_batch)
            train_data_images_batches.append(training_data_batch[b'data'])
            train_data_labels_batches.append(training_data_batch[b'labels'])

        full_training_images = np.vstack(train_data_images_batches)
        full_training_labels = np.hstack(train_data_labels_batches)

        validation_data_indexes = np.random.choice(range(full_training_images.shape[0]), 5000, replace=False)

        validation_data_images = full_training_images[validation_data_indexes, :]
        validation_data_labels = full_training_labels[validation_data_indexes]

        full_training_images = np.delete(full_training_images, validation_data_indexes, axis=0)
        full_training_labels = np.delete(full_training_labels, validation_data_indexes)

        testing_data = unpickle(file=f'Datasets/{testkey}')
        testing_data_images = testing_data[b'data']
        testing_data_labels = testing_data[b'labels']

        train_data_mean = np.mean(full_training_images, axis=0)
        train_data_std = np.std(full_training_images, axis=0)

        train_data_images = (full_training_images - train_data_mean) / train_data_std
        validation_data_images = (validation_data_images - train_data_mean) / train_data_std
        test_data_images = (testing_data_images - train_data_mean) / train_data_std

        return {'training_data': {'images': train_data_images, 'labels': full_training_labels},
                'validation_data': {'images': validation_data_images, 'labels': validation_data_labels},
                'testing_data': {'images': test_data_images, 'labels': np.array(testing_data_labels)},
                'mean_normalization': train_data_mean,
                'std_normalization': train_data_std}


def VIT_dataprepocessing_model_phase(
        image_size: int = 32,
        patch_qty: int = 16,
        size_per_patch: int = 16,
        dimension_dense_projection: int = 256,
        dense_activation: str = "relu"
) -> Model:

    input_dimensionality = 1 + patch_qty
    inputs = Input(shape=(image_size,
                          image_size,
                          3))

    patches = Reshape((patch_qty,  # create some patches
                      size_per_patch,
                      size_per_patch,
                      3))(inputs)

    flattened_patches = Reshape((patch_qty,  # flatten them
                                 size_per_patch * size_per_patch * 3))(patches)

    projection = Dense(units=dimension_dense_projection,  # project/embedd them nonlinearly
                       activation=dense_activation)(flattened_patches)

    batch_size = projection.shape[0]  # insert cls-token at first index
    cls = np.ones((batch_size,
                   1,
                   dimension_dense_projection))

    sequential_vectors_non_processed = tf.stack([cls,
                                           projection], axis=1)
    sequential_vectors = tf.reshape(sequential_vectors_non_processed,
                                    [-1, patch_qty + 1, dimension_dense_projection])

    positions = tf.expand_dims(tf.range(patch_qty + 1, dtype=tf.float32), axis=0)
    # add positional embeddings
    positional_embeddings = layers.Embedding(input_dim=input_dimensionality,
                                             output_dim=dimension_dense_projection)(positions)

    return Model(inputs=inputs,
                 outputs=positional_embeddings + sequential_vectors)