import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Reshape, Input, Dense
import numpy as np
import pickle
from typing import Dict, Union, Any


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
    x_test,
    y_test,
    image_size: int = 32,
    patch_qty: int = 16,
    size_per_patch: int = 8,
    dimension_dense_projection: int = 240,
    dense_activation: str = "gelu",
    data_inputs=None,
    data_outputs=10,
    image_height=16,
    image_width=16,
) -> Model:

    input_dimensionality = 1 + patch_qty
    inputs = Input(shape=(image_size, image_size, 3), batch_size=None)

    patches = Reshape((patch_qty, size_per_patch, size_per_patch, 3))(inputs)

    flattened_patches = Reshape(
        (patch_qty, size_per_patch * size_per_patch * 3)
    )(patches)

    projection = Dense(
        units=dimension_dense_projection, activation=dense_activation
    )(flattened_patches)

    batch_size = x_train.shape[1]

    sequential_vectors_non_processed = CLSTokenLayer(dimension_dense_projection)(projection)

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

    encoded_layer = PositionalEmbedding()(sequential_vectors)
    encoder_output = encoded_layer

    n_encoder_blocks = 12
    for i in range(n_encoder_blocks):
        encoder_output = EncoderBlock(12, 4 * dimension_dense_projection, dropout=0.1)(encoder_output)

    outputs = layers.Softmax(axis=1)(layers.Dense(units=data_outputs, name='head')(encoder_output[:, 0]))
    model = Model(inputs, outputs)

    print(model.summary())

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    stats = model.fit(train_data, validation_data=valid_data, epochs=50, batch_size=batch_size)

    preds = model.predict(valid_data)

    evaluated = model.evaluate(valid_data, verbose=2)

    print(stats)
    print(evaluated)
    # return inputs, positional_embeddings + sequential_vectors
    # return Model(
    #     inputs=inputs, outputs=positional_embeddings + sequential_vectors
    # )

import keras.activations
import tensorflow as tf
from keras import layers, Model
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

    def call(self, inputs, *args, **kwargs):
        # Did not implement a mask - Don't know if it is required

        # Compute the Q, K, V matrices as given in attention is all you need
        Query = tf.matmul(inputs, self.W_query)
        Key = tf.matmul(inputs, self.W_key)
        Value = tf.matmul(inputs, self.W_value)
        # Calculate similarity measure and apply softmax along columns
        similarity = tf.matmul(Query, tf.transpose(Key, perm=[0, 2, 1]))
        similarity = tf.nn.softmax(tf.math.divide(similarity, tf.cast(self.key_size, dtype=inputs.dtype) ** 0.5), axis=2)
        attention_matrix = tf.matmul(similarity, Value)
        return attention_matrix


class MultiHeadAttention(Layer):
    def __init__(self, n_channels):
        super(MultiHeadAttention, self).__init__()
        self.n_channels = n_channels
        self.linear_transform = None
        self.attention_layers = []

    def build(self, input_shape):
        if input_shape[-1] % self.n_channels != 0:
            print("The embedding size needs to be divisible by the number of channels!")
            raise Exception
        reduced_embedding = int(input_shape[-1] / self.n_channels)
        # Create stack of Attention layers
        for i in range(self.n_channels):
            self.attention_layers.append(DotProductAttention(reduced_embedding, reduced_embedding, reduced_embedding))
        # Initialize the last linear transform
        initializer = tf.initializers.GlorotNormal()
        self.linear_transform = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=initializer,
                                       trainable=True)

    def call(self, inputs, *args, **kwargs):
        concatenated_result = []
        for i in range(self.n_channels):
            # Each attention layer outputs an array with n_embedding in size, which concatenates to the full input shape
            attention_output = self.attention_layers[i](inputs)

            # Store the results and then concatenate them
            concatenated_result.append(attention_output)
        # Concatenate the results
        concatenated_result = tf.concat(concatenated_result, axis=2)
        # Perform the final linear transform
        output = tf.matmul(concatenated_result, self.linear_transform)
        return output


class NormLayer(Layer):
    def __init__(self, epsilon=10 ** (-8)):
        super(NormLayer, self).__init__()
        self.scale = None
        self.shift = None
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],), initializer=tf.keras.initializers.Ones(), trainable=True)
        self.shift = self.add_weight(shape=(input_shape[-1],), initializer=tf.keras.initializers.Zeros(), trainable=True)

    def call(self, inputs, *args, **kwargs):
        # Normalize over hidden layers instead of training samples
        mean = tf.reduce_mean(inputs, axis=2, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=2, keepdims=True)
        x_norm = (inputs - mean) / (std + self.epsilon)
        outputs = tf.multiply(x_norm, self.scale) + self.shift
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

    def call(self, inputs, *args, **kwargs):
        # Normalize over hidden layers instead of training samples
        training = kwargs.get('training', False)
        outputs = tf.matmul(inputs, self.W1) + self.b1
        outputs = keras.activations.gelu(outputs)
        outputs = keras.layers.Dropout(self.dropout, outputs.shape)(outputs, training)
        outputs = tf.matmul(outputs, self.W2) + self.b2
        outputs = keras.layers.Dropout(self.dropout, outputs.shape)(outputs, training)
        return outputs


class EncoderBlock(Layer):
    def __init__(self, n_channels, hidden_nodes, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.MLA = MultiHeadAttention(n_channels)
        self.MLP = MLP(hidden_nodes, dropout)
        self.LayerNorm1 = NormLayer()
        self.LayerNorm2 = NormLayer()

    def call(self, inputs, *args, **kwargs):
        initial_inputs = inputs
        normalized_inputs = self.LayerNorm1(inputs)
        attention_output = self.MLA(normalized_inputs)
        combined_output = attention_output + initial_inputs
        normalized_inputs = self.LayerNorm2(combined_output)
        feed_forward_output = self.MLP(normalized_inputs)
        encoder_output = feed_forward_output + combined_output
        return encoder_output


class CLSTokenLayer(Layer):
    def __init__(self, d_model):
        super(CLSTokenLayer, self).__init__()
        self.cls_token = self.add_weight(shape=(1, 1, d_model), initializer='zeros', trainable=True)

    def call(self, inputs, *args, **kwargs):
        # Repeat the cls token across the batch and concatenate with inputs
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        return tf.concat([cls_tokens, inputs], axis=1)


class PositionalEmbedding(Layer):
    def __init__(self,):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = None

    def build(self, input_shape):
        initializer = tf.initializers.GlorotNormal()
        self.pos_embedding = self.add_weight(shape=(1, input_shape[1], input_shape[-1]), initializer=initializer)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.pos_embedding
