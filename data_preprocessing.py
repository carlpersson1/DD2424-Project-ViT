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

        training_data = get_batches(**{'data': train_data_images, 'labels': full_training_labels, 'batch_size': 32})
        validation_data = get_batches(**{'data': validation_data_images, 'labels': validation_data_labels, 'batch_size': 32})
        testing_data = get_batches(**{'data': test_data_images, 'labels': np.array(testing_data_labels), 'batch_size': 32})

        training_data_tbret = {'images': [],
                               'labels': []}
        validation_data_tbret = {'images': [],
                                 'labels': []}
        testing_data_tbret = {'images': [],
                              'labels': []}

        for entry in training_data:
            training_data_tbret['images'].append(entry['images'].reshape(32, 32, 32, 3))
            training_data_tbret['labels'].append(entry['labels'])

        for entry in validation_data:
            validation_data_tbret['images'].append(entry['images'].reshape(32, 32, 32, 3))
            validation_data_tbret['labels'].append(entry['labels'])

        for entry in testing_data:
            testing_data_tbret['images'].append(entry['images'].reshape(32, 32, 32, 3))
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
    x_test,
    y_test,
    image_size: int = 32,
    patch_qty: int = 32,
    size_per_patch: int = 8,
    dimension_dense_projection: int = 256,
    dense_activation: str = "relu",
    data_inputs=None,
    data_outputs=10,
    image_height=16,
    image_width=16,
) -> Model:

    input_dimensionality = 1 + patch_qty
    inputs = Input(shape=(image_size, image_size, 3), batch_size=32)

    patch_qty = 16
    patches = Reshape((patch_qty, size_per_patch, size_per_patch, 3))(inputs)

    flattened_patches = Reshape(
        (patch_qty, size_per_patch * size_per_patch * 3)
    )(patches)

    projection = Dense(
        units=dimension_dense_projection, activation=dense_activation
    )(flattened_patches)

    batch_size = x_train.shape[1]
    cls = np.ones(
        (batch_size, 1, dimension_dense_projection), dtype="float32"
    )
    cls = tf.convert_to_tensor(cls)  # Convert to TensorFlow tensor

    projection = tf.convert_to_tensor(projection)  # Convert to TensorFlow tensor

    sequential_vectors_non_processed = tf.concat(
        [cls, projection], axis=1
    )
    sequential_vectors = tf.reshape(
        sequential_vectors_non_processed,
        [-1, patch_qty + 1, dimension_dense_projection],
    )

    positions = tf.expand_dims(
        tf.range(patch_qty + 1, dtype=tf.float32), axis=0
    )
    positional_embeddings = layers.Embedding(
        input_dim=input_dimensionality, output_dim=dimension_dense_projection
    )(positions)

    encoded_layer = positional_embeddings + sequential_vectors

    attentionlayer = MultiHeadAttention(17)

    normlayer = NormLayer()

    layernormalized = normlayer(encoded_layer)

    transformerprocessed = attentionlayer(layernormalized)

    transformerprocessed = layers.Add(transformerprocessed, encoded_layer)

    layernormalized2 = NormLayer(transformerprocessed)

    mlpprocessed = MLP(100, dropout=0.0)(layernormalized2)

    processed = layers.Add(mlpprocessed, transformerprocessed)

    outputs = layers.Dense(units=len(data_outputs),
                           name='head',
                           kernel_initializer=tf.keras.initializers.zeros)(processed)

    model = Model(inputs, outputs)

    print(model.summary())

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)
    # return inputs, positional_embeddings + sequential_vectors
    # return Model(
    #     inputs=inputs, outputs=positional_embeddings + sequential_vectors
    # )
import keras.activations
import tensorflow as tf
from keras import layers, Model
from keras.initializers import GlorotNormal
from keras.layers import Reshape, Input, Dense, Layer
import numpy as np
import pickle
from typing import Dict, Union, Any


class DotProductAttention(Layer):
    def __init__(self, query_size, key_size, value_size):
        super(DotProductAttention, self).__init__()
        self.W_query = None
        self.W_key = None
        self.W_value = None

        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

    def build(self, input_shape):
        # Called only once to create state of layer - Can be used to dynamically set input sizes
        # Which initializer to use? Chose Xavier for the time being
        initializer = tf.initializers.GlorotNormal()
        self.W_query = self.add_weight(shape=(input_shape[-1], self.query_size), initializer=initializer,
                                       trainable=True)
        self.W_key = self.add_weight(shape=(input_shape[-1], self.key_size), initializer=initializer,
                                       trainable=True)
        self.W_value = self.add_weight(shape=(input_shape[-1], self.value_size), initializer=initializer,
                                       trainable=True)

    def call(self, inputs):
        # Did not implement a mask - Don't know if it is required

        # Compute the Q, K, V matrices as given in attention is all you need
        Query = tf.matmul(inputs, self.W_query)
        Key = tf.matmul(inputs, self.W_key)
        Value = tf.matmul(inputs, self.W_value)

        # Calculate similarity measure and apply softmax along columns
        similarity = tf.nn.softmax(tf.matmul(Query, tf.transpose(Key)), axis=1)
        attention_matrix = tf.matmul(tf.math.divide(similarity, self.key_size ** 0.5), Value)
        return attention_matrix


class MultiHeadAttention(Layer):
    def __init__(self, n_channels):
        super(MultiHeadAttention, self).__init__()
        self.n_channels = n_channels
        self.linear_transform = None
        self.attention_layers = []

    def build(self, input_shape):
        if input_shape[1] % self.n_channels != 0:
            print("The embedding size needs to be divisible by the number of channels!")
            raise Exception
        reduced_embedding = int(input_shape[1] / self.n_channels)

        # Create stack of Attention layers
        for i in range(self.n_channels):
            self.attention_layers.append(DotProductAttention(reduced_embedding, reduced_embedding, reduced_embedding))

        # Initialize the last linear transform
        initializer = tf.initializers.GlorotNormal()
        self.linear_transform = self.add_weight(shape=(input_shape[1], input_shape[1]), initializer=initializer,
                                       trainable=True)

    def call(self, inputs):
        concatenated_result = 0
        for i in range(self.n_channels):
            # Each attention layer outputs an array with n_embedding in size, which concatenates to the full input shape
            attention_output = self.attention_layers[i](inputs)

            # Concatenate the result
            if i > 0:
                concatenated_result = tf.concat([concatenated_result, attention_output], axis=1)
            else:
                concatenated_result = attention_output

        # Perform the final linear transform
        output = tf.matmul(concatenated_result, self.linear_transform)

        return output


class NormLayer(Layer):
    def __init__(self):
        super(NormLayer, self).__init__()

    def call(self, inputs, epsilon=10 ** (-8)):
        # Normalize over hidden layers instead of training samples
        mean = tf.reduce_mean(inputs, axis=1)
        std = tf.math.reduce_std(inputs, axis=1)
        outputs = (inputs - mean[:, tf.newaxis]) / (std[:, tf.newaxis] + epsilon)
        return outputs


class MLP(Layer):
    def __init__(self, hidden_units, dropout=0.0):
        super(MLP, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def build(self, input_shape):
        # Called only once to create state of layer - Can be used to dynamically set input sizes
        # Which initializer to use? Chose Xavier for the time being
        initializer = tf.initializers.GlorotNormal()
        self.W1 = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer=initializer,
                                       trainable=True)
        self.W2 = self.add_weight(shape=(self.hidden_units, input_shape[-1]), initializer=initializer,
                                  trainable=True)
        self.b1 = self.add_weight(shape=(self.hidden_units,), initializer=initializer, trainable=True)
        self.b2 = self.add_weight(shape=(input_shape[-1],), initializer=initializer, trainable=True)

    def call(self, inputs, training):
        # Normalize over hidden layers instead of training samples
        outputs = tf.matmul(inputs, self.W1) + self.b1
        outputs = keras.activations.gelu(outputs)
        outputs = keras.layers.Dropout(self.dropout, outputs.shape)(outputs, training)
        outputs = tf.matmul(outputs, self.W2) + self.b2
        outputs = keras.layers.Dropout(self.dropout, outputs.shape)(outputs, training)
        return outputs
