import tensorflow as tf
from keras import layers, Model
from keras.initializers.initializers import GlorotNormal
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
    def __init__(self, n_channels, query_size, key_size):
        super(MultiHeadAttention, self).__init__()
        self.n_channels = n_channels
        self.query_size = query_size
        self.key_size = key_size
        self.linear_transform = None
        self.reduced_embedding = None
        self.attention_layers = []

    def build(self, input_shape):
        if input_shape[1] % self.n_channels != 0:
            print("The embedding size needs to be divisible by the number of channels!")
            raise Exception
        self.reduced_embedding = int(input_shape[1] / self.n_channels)

        # Create stack of Attention layers
        for i in range(self.n_channels):
            self.attention_layers.append(DotProductAttention(self.query_size, self.key_size, self.reduced_embedding))

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


if __name__ == "__main__":
    attention = MultiHeadAttention(4, 4, 4)
    test = tf.constant([[10.0, 1.4, 1.3, 4.2], [-5.4, 3.2, 1.7, 3.2], [1.6, 5.1, 6.3, 1.2]])

    norm = NormLayer()
    print(norm(test))
    print(attention(test))

