import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Reshape, Input, Dense
import numpy as np
import pickle
from typing import Dict, Union, Any
from Multihead_attention import *

def get_cifar_dataset_scaled(
        batch_size = 32,
        key: str = "data_batch",
        testkey: str = "test_batch"
) -> Dict[str, Union[Union[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Any]]:
        def get_batches(data: np.ndarray, labels: np.ndarray, batch_size: int):
            num_samples = data.shape[0]
            num_batches = num_samples // batch_size
            batches = []

            for i in range(num_batches):
                batch_data = data[i * batch_size: (i + 1) * batch_size]
                batch_labels = labels[i * batch_size: (i + 1) * batch_size]
                batches.append({'images': batch_data, 'labels': batch_labels})

            return batches

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

        training_data = get_batches(**{'data': train_data_images, 'labels': full_training_labels, 'batch_size': batch_size})
        validation_data = get_batches(**{'data': validation_data_images, 'labels': validation_data_labels, 'batch_size': batch_size})
        testing_data = get_batches(**{'data': test_data_images, 'labels': np.array(testing_data_labels), 'batch_size': batch_size})

        training_data_tbret = {'images': [],
                               'labels': []}
        validation_data_tbret = {'images': [],
                                 'labels': []}
        testing_data_tbret = {'images': [],
                              'labels': []}

        for entry in training_data:
            training_data_tbret['images'].append(entry['images'].reshape(batch_size, 32, 32, 3))
            training_data_tbret['labels'].append(entry['labels'])

        for entry in validation_data:
            validation_data_tbret['images'].append(entry['images'].reshape(batch_size, 32, 32, 3))
            validation_data_tbret['labels'].append(entry['labels'])

        for entry in testing_data:
            testing_data_tbret['images'].append(entry['images'].reshape(batch_size, 32, 32, 3))
            testing_data_tbret['labels'].append(entry['labels'])

        return {'training_data': training_data_tbret,
                'validation_data': validation_data_tbret,
                'testing_data': testing_data_tbret,
                'mean_normalization': train_data_mean,
                'std_normalization': train_data_std}
       #  print("-")
        # tbret = {'training_data': get_batches(**{'data': train_data_images, 'labels': full_training_labels, 'batch_size': 32}),
        #         'validation_data': get_batches(**{'data': validation_data_images, 'labels': validation_data_labels, 'batch_size': 32}),
        #         'testing_data': get_batches(**{'data': test_data_images, 'labels': np.array(testing_data_labels), 'batch_size': 32}),
        #         'mean_normalization': train_data_mean,
        #         'std_normalization': train_data_std}

     #    print("-")

def VIT_dataprepocessing_model_phase(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    trial=None,
    monitor=None,
    image_size: int = 32,
    n_channels_encoder_block=12,
    dropout_encoder_block=0.1,
    L2_reg_encoder_block=0.001,
    gauss_noise=0.0,
    patch_qty: int = 16,
    n_encoder_blocks=12,
    epochs=5,
    lr=0.001,
    dimension_dense_projection: int = 240,
    size_per_patch: int = 8,
    dense_activation: str = "linear",
    data_outputs=10
) -> Model:

    inputs = Input(shape=(image_size, image_size, 3), batch_size=None)

    # Apply gaussian noise and image flipping to increase diversity of inputs
    modified_inputs = TrainingModifier(gauss_noise)(inputs)

    # Extract patches properly
    patches = tf.image.extract_patches(
        images=modified_inputs,
        sizes=[1, size_per_patch, size_per_patch, 1],
        strides=[1, size_per_patch, size_per_patch, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    flattened_patches = Reshape(
        (patch_qty, size_per_patch * size_per_patch * 3)
    )(patches)

    projection = Dense(
        units=dimension_dense_projection, activation=dense_activation
    )(flattened_patches)

    projection = CLSTokenLayer(dimension_dense_projection)(projection)

    encoded_layer = PositionalEmbedding()(projection)

    encoded_layer = tf.keras.layers.Dropout(dropout_encoder_block)(encoded_layer)
    encoder_output = encoded_layer

    for i in range(n_encoder_blocks):
        encoder_output = EncoderBlock(n_channels_encoder_block,
                                      4 * dimension_dense_projection,
                                      dropout=dropout_encoder_block,
                                      L2_reg=L2_reg_encoder_block)(encoder_output)

    outputs = layers.Softmax(axis=1)(layers.Dense(units=data_outputs, name='head')(encoder_output[:, 0]))
    model = Model(inputs, outputs)

    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    testing_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = x_train.shape[1]
    if trial is not None:
        from optuna.integration import TFKerasPruningCallback
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            TFKerasPruningCallback(trial, monitor),
        ]

        stats = model.fit(train_data,
                          validation_data=valid_data,
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=callbacks)
    else:
        stats = model.fit(train_data,
                          validation_data=valid_data,
                          epochs=epochs,
                          batch_size=batch_size)

    evaluation_results = model.evaluate(valid_data, verbose=2)

    testing_results = model.evaluate(testing_data, verbose=2)
    print(stats)
    print(evaluation_results)
    print(testing_results)

    return stats, evaluation_results, testing_results


def ViT_Hybrid_Architecture(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    trial=None,
    monitor=None,
    image_size: int = 32,
    n_channels_encoder_block=12,
    dropout_encoder_block=0.1,
    L2_reg_encoder_block=0.001,
    gauss_noise=0.0,
    patch_qty: int = 16,
    n_encoder_blocks=12,
    kernel_size=(3, 3),
    epochs=5,
    lr=0.001,
    dimension_dense_projection: int = 240,
    size_per_patch: int = 8,
    dense_activation: str = "linear",
    data_outputs=10
) -> Model:

    inputs = Input(shape=(image_size, image_size, 3), batch_size=None)

    # Apply gaussian noise and image flipping to improve diversity of inputs
    modified_inputs = TrainingModifier(gauss_noise)(inputs)

    # Patch embedding through a regular convolutional layer
    convolutional_patches = tf.keras.layers.Convolution2D(dimension_dense_projection,
                                                          kernel_size=size_per_patch,
                                                          strides=size_per_patch)(modified_inputs)

    convmixer = ConvMixer(dimension_dense_projection, kernel_size)(convolutional_patches)

    # Flatten the output of the convmixer
    flattened_patches = Reshape(
        (patch_qty, dimension_dense_projection)
    )(convmixer)

    # Maybe not necessary

    projection = Dense(
        units=dimension_dense_projection, activation=dense_activation
    )(flattened_patches)


    projection = CLSTokenLayer(dimension_dense_projection)(flattened_patches)

    encoded_layer = PositionalEmbedding()(projection)

    encoded_layer = tf.keras.layers.Dropout(dropout_encoder_block)(encoded_layer)
    encoder_output = encoded_layer

    for i in range(n_encoder_blocks):
        encoder_output = EncoderBlock(n_channels_encoder_block,
                                      4 * dimension_dense_projection,
                                      dropout=dropout_encoder_block,
                                      L2_reg=L2_reg_encoder_block)(encoder_output)

    outputs = layers.Softmax(axis=1)(layers.Dense(units=data_outputs, name='head')(encoder_output[:, 0]))
    model = Model(inputs, outputs)

    print(model.summary())

    batch_size = x_train.shape[1]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    testing_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    stats = model.fit(train_data,
                      validation_data=valid_data,
                      epochs=epochs,
                      batch_size=batch_size)

    evaluation_results = model.evaluate(valid_data, verbose=2)

    testing_results = model.evaluate(testing_data, verbose=2)
    print(stats)
    print(evaluation_results)
    print(testing_results)

    return stats, evaluation_results, testing_results
