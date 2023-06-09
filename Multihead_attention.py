import keras.activations
import tensorflow as tf
from keras import layers, Model
from keras.layers import Layer, Reshape, Input, Dense


class DotProductAttention(Layer):
    def __init__(self, query_size, key_size, value_size, L2_reg=0.001):
        super(DotProductAttention, self).__init__()
        self.L2_reg = L2_reg
        self.W_query = None
        self.W_key = None
        self.W_value = None

        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

    def build(self, input_shape):
        # Called only once to create state of layer - Can be used to dynamically set input sizes
        # Which initializer to use? Chose Xavier for the time being
        initializer = tf.initializers.GlorotUniform()
        self.W_query = self.add_weight(shape=(input_shape[-1], self.query_size), initializer=initializer,
                                       trainable=True, regularizer=tf.keras.regularizers.L2(self.L2_reg))
        self.W_key = self.add_weight(shape=(input_shape[-1], self.key_size), initializer=initializer,
                                     trainable=True, regularizer=tf.keras.regularizers.L2(self.L2_reg))
        self.W_value = self.add_weight(shape=(input_shape[-1], self.value_size), initializer=initializer,
                                       trainable=True, regularizer=tf.keras.regularizers.L2(self.L2_reg))

    def call(self, inputs, *args, **kwargs):
        # Did not implement a mask - Don't know if it is required

        # Compute the Q, K, V matrices as given in attention is all you need
        query = tf.matmul(inputs, self.W_query)
        key = tf.matmul(inputs, self.W_key)
        value = tf.matmul(inputs, self.W_value)

        # Calculate similarity measure and apply softmax along columns
        similarity = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))
        similarity = tf.nn.softmax(tf.math.divide(similarity, tf.cast(self.key_size,
                                                                      dtype=inputs.dtype) ** 0.5), axis=-1)
        attention_matrix = tf.matmul(similarity, value)
        return attention_matrix


class MultiHeadAttention(Layer):
    def __init__(self, n_channels, L2_reg=0.001):
        super(MultiHeadAttention, self).__init__()
        self.n_channels = n_channels
        self.linear_transform = None
        self.attention_layers = []
        self.L2_reg = L2_reg

    def build(self, input_shape):
        if input_shape[-1] % self.n_channels != 0:
            print("The embedding size needs to be divisible by the number of channels!")
            raise Exception
        reduced_embedding = int(input_shape[-1] / self.n_channels)
        # Create stack of Attention layers
        for i in range(self.n_channels):
            self.attention_layers.append(DotProductAttention(reduced_embedding, reduced_embedding,
                                                             reduced_embedding, self.L2_reg))
        # Initialize the last linear transform
        initializer = tf.initializers.GlorotUniform()
        self.linear_transform = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=initializer,
                                                trainable=True, regularizer=tf.keras.regularizers.L2(self.L2_reg))

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
        self.gamma = None
        self.beta = None
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],), initializer=tf.keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1],), initializer=tf.keras.initializers.Zeros(), trainable=True)

    def call(self, inputs, *args, **kwargs):
        # Normalize over hidden layers instead of training samples
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=-1, keepdims=True)
        x_norm = (inputs - mean) / (std + self.epsilon)
        outputs = tf.multiply(x_norm, self.gamma) + self.beta
        return outputs


class MLP(Layer):
    def __init__(self, hidden_units, dropout=0.0, L2_reg=0.001):
        super(MLP, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.L2_reg = L2_reg
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def build(self, input_shape):
        # Called only once to create state of layer - Can be used to dynamically set input sizes
        # Which initializer to use? Chose Xavier for the time being
        initializer = tf.initializers.GlorotUniform()
        self.W1 = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer=initializer,
                                  trainable=True, regularizer=tf.keras.regularizers.L2(self.L2_reg))
        self.W2 = self.add_weight(shape=(self.hidden_units, input_shape[-1]), initializer=initializer,
                                  trainable=True, regularizer=tf.keras.regularizers.L2(self.L2_reg))
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
    def __init__(self, n_channels, hidden_nodes, dropout=0.0, L2_reg=0.001):
        super(EncoderBlock, self).__init__()
        self.MLA = MultiHeadAttention(n_channels, L2_reg)
        self.MLP = MLP(hidden_nodes, dropout, L2_reg)
        self.LayerNorm1 = NormLayer()
        self.LayerNorm2 = NormLayer()
        self.drop_layer = keras.layers.Dropout(dropout)

    def call(self, inputs, *args, **kwargs):
        # Layers combined as described in 'An Image is worth 16x16 words' to form an encoder
        normalized_inputs = self.LayerNorm1(inputs)
        attention_output = self.MLA(normalized_inputs)
        attention_output = self.drop_layer(attention_output)
        combined_output = attention_output + inputs
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
        initializer = tf.initializers.GlorotUniform()
        self.pos_embedding = self.add_weight(shape=(1, input_shape[1], input_shape[-1]), initializer=initializer)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.pos_embedding


class TrainingModifier(Layer):
    def __init__(self, noise):
        super(TrainingModifier, self).__init__()
        self.flip = tf.keras.layers.RandomFlip(mode='horizontal')
        self.gauss = tf.keras.layers.GaussianNoise(noise)

    def call(self, inputs, *args, **kwargs):
        inputs = self.flip(inputs)
        outputs = self.gauss(inputs)
        return outputs


class ConvMixer(Layer):
    def __init__(self, filters, kernel_size):
        super(ConvMixer, self).__init__()
        # Depth-wise convolution
        self.depth_conv = keras.layers.convolutional.DepthwiseConv2D(kernel_size=kernel_size,
                                                                     padding='same', trainable=True)
        # Point-wise convolution
        self.point_conv = keras.layers.convolutional.Convolution2D(filters, (1, 1), trainable=True)
        self.GELU = keras.activations.gelu
        self.batch_norm1 = keras.layers.BatchNormalizationV2(trainable=True)
        self.batch_norm2 = keras.layers.BatchNormalizationV2(trainable=True)

    def call(self, inputs, *args, **kwargs):
        depth_conv = self.depth_conv(inputs)
        depth_out = self.GELU(depth_conv)
        depth_out = self.batch_norm1(depth_out)
        mid_point = depth_out + inputs
        point_conv = self.point_conv(mid_point)
        point_out = self.GELU(point_conv)
        output = self.batch_norm2(point_out)
        return output

