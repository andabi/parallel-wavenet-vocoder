# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.signal import stft


def causal_conv(value, filter_, dilation, name='causal_conv'):
    def time_to_batch(value, dilation):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

    def batch_to_time(value, dilation):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])

    with tf.variable_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            # for left-side padding because tf.nn.conv1d do not support left-side padding with padding='SAME'
            padded = tf.pad(transformed, [[0, 0], [filter_width - 1, 0], [0, 0]])
            conv = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)

            # Remove excess elements at the end caused by padding in time_to_batch.
            result = tf.slice(restored,
                              [0, 0, 0],
                              [-1, tf.shape(value)[1], -1])
        else:
            padded = tf.pad(value, [[0, 0], [filter_width - 1, 0], [0, 0]])
            result = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
        return result


# Renovated based on https://github.com/ibab/tensorflow-wavenet
class WaveNet(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2 ** 8,
                 use_biases=False,
                 condition_channels=None,
                 use_skip_connection=True,
                 normalize=None,
                 is_training=True,
                 name='wavenet'):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.condition_channels = condition_channels
        self.use_skip_connection = use_skip_connection
        self.normalize = normalize
        self.is_training = is_training
        self.name = name

    # network
    def __call__(self, input_batch, condition_batch=None):
        with tf.variable_scope(self.name):
            '''Construct the WaveNet network.'''
            outputs = []
            with tf.variable_scope('causal_layer'):
                current_layer = self._create_causal_layer(input_batch)

            # Add all defined dilation layers.
            with tf.variable_scope('dilated_stack'):
                for layer_index, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(layer_index)):
                        output, current_layer = self._create_dilation_layer(
                            current_layer, dilation, condition_batch)
                        outputs.append(output)

            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to postprocess the output.
            with tf.variable_scope('postprocessing'):
                # We skip connections from the outputs of each layer, adding them all up here.
                total = sum(outputs) if self.use_skip_connection else outputs[-1]
                transformed1 = tf.nn.relu(total)
                if self.normalize:
                    transformed1 = normalize(transformed1, method=self.normalize, is_training=self.is_training,
                                             name='normalize_postprocess1')
                w1 = tf.get_variable('postprocess1', [1, self.skip_channels, self.skip_channels])
                conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
                if self.use_biases:
                    b1 = tf.get_variable('postprocess1_bias', [self.skip_channels], initializer=tf.zeros_initializer)
                    conv1 = tf.add(conv1, b1)
                transformed2 = tf.nn.relu(conv1)
                if self.normalize:
                    transformed2 = normalize(transformed2, method=self.normalize, is_training=self.is_training,
                                             name='normalize_postprocess2')
                w2 = tf.get_variable('postprocess2', [1, self.skip_channels, self.quantization_channels])
                conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
                if self.use_biases:
                    b2 = tf.get_variable('postprocess2_bias', [self.quantization_channels], initializer=tf.zeros_initializer)
                    conv2 = tf.add(conv2, b2)
        return conv2

    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        receptive_field += filter_width - 1
        return receptive_field

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        weights_filter = tf.get_variable('filter', [self.filter_width, self.quantization_channels, self.residual_channels])
        layer = causal_conv(input_batch, weights_filter, 1)
        if self.normalize:
            layer = normalize(layer, method=self.normalize, is_training=self.is_training)
        return layer

    def _create_dilation_layer(self, input_batch, dilation, condition_batch):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             conditioning_batch: Tensor containing the global or local data upon
                 which the output is to be conditioned upon. Shape:
                 In global case, shape=[batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.
                 In local case, shape=[batch size, n_timesteps, channels].

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and conditioning
        are omitted due to the limits of ASCII art.

        '''

        weights_filter = tf.get_variable('filter', [self.filter_width, self.residual_channels, self.dilation_channels])
        weights_gate = tf.get_variable('gate', [self.filter_width, self.residual_channels, self.dilation_channels])

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if condition_batch is not None:
            weights_cond_filter = tf.get_variable('gc_filter', [1, self.condition_channels, self.dilation_channels])
            conv_filter = conv_filter + tf.nn.conv1d(condition_batch, weights_cond_filter, stride=1, padding="SAME",
                                                     name="gc_filter")
            weights_cond_gate = tf.get_variable('gc_gate', [1, self.condition_channels, self.dilation_channels])
            conv_gate = conv_gate + tf.nn.conv1d(condition_batch, weights_cond_gate, stride=1, padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = tf.get_variable('filter_bias', [self.dilation_channels], initializer=tf.zeros_initializer)
            gate_bias = tf.get_variable('gate_bias', [self.dilation_channels], initializer=tf.zeros_initializer)
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        if self.normalize:
            conv_filter = normalize(conv_filter, method=self.normalize, is_training=self.is_training,
                                    name='normalize_filter')
            conv_gate = normalize(conv_gate, method=self.normalize, is_training=self.is_training,
                                  name='normalize_gate')

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = tf.get_variable('dense', [1, self.dilation_channels, self.residual_channels])
        transformed = tf.nn.conv1d(out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        weights_skip = tf.get_variable('skip', [1, self.dilation_channels, self.skip_channels])
        skip_output = tf.nn.conv1d(out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = tf.get_variable('dense_bias', [self.residual_channels], initializer=tf.zeros_initializer)
            skip_bias = tf.get_variable('skip_bias', [self.skip_channels], initializer=tf.zeros_initializer)
            transformed = transformed + dense_bias
            skip_output = skip_output + skip_bias
        dense_output = input_batch + transformed

        if self.normalize:
            skip_output = normalize(skip_output, method=self.normalize, is_training=self.is_training,
                                    name='normalize_skip_output')
            dense_output = normalize(dense_output, method=self.normalize, is_training=self.is_training,
                                     name='normalize_dense_output')

        return skip_output, dense_output


# TODO normalization refactoring
def normalize(input, is_training, method='bn', name='normalize'):
    with tf.variable_scope(name):
        if method == 'bn':
            input = tf.layers.batch_normalization(input, training=is_training)
        elif method == 'in':
            input = instance_normalization(input)
            # elif hp.model.normalize == 'wn':
    return input


# TODO generalization
def instance_normalization(input, epsilon=1e-8):
    inputs_shape = input.get_shape()
    params_shape = inputs_shape[-1:]
    time_axis = 1

    mean, variance = tf.nn.moments(input, [time_axis], keep_dims=True)
    beta = tf.get_variable("beta", shape=params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer)
    normalized = (input - mean) / ((variance + epsilon) ** (.5))
    output = gamma * normalized + beta
    return output


# def get_var_maybe_avg(var_name, ema, **kwargs):
#     ''' utility for retrieving polyak averaged params '''
#     v = tf.get_variable(var_name, **kwargs)
#     if ema is not None:
#         v = ema.average(v)
#     return v
#
# def get_vars_maybe_avg(var_names, ema, **kwargs):
#     ''' utility for retrieving polyak averaged params '''
#     vars = []
#     for vn in var_names:
#         vars.append(get_var_maybe_avg(vn, ema, **kwargs))
#     return vars
#
# def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' fully connected layer '''
#     if init:
#         # data based initialization of parameters
#         V = tf.get_variable('V', [int(x.get_shape()[1]),num_units], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
#         V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
#         x_init = tf.matmul(x, V_norm)
#         m_init, v_init = tf.nn.moments(x_init, [0])
#         scale_init = init_scale/tf.sqrt(v_init + 1e-10)
#         g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
#         b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
#         x_init = tf.reshape(scale_init,[1,num_units])*(x_init-tf.reshape(m_init,[1,num_units]))
#         if nonlinearity is not None:
#             x_init = nonlinearity(x_init)
#         return x_init
#
#     else:
#         V,g,b = get_vars_maybe_avg(['V','g','b'], ema)
#         tf.assert_variables_initialized([V,g,b])
#
#         # use weight normalization (Salimans & Kingma, 2016)
#         x = tf.matmul(x, V)
#         scaler = g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))
#         x = tf.reshape(scaler,[1,num_units])*x + tf.reshape(b,[1,num_units])
#
#         # apply nonlinearity
#         if nonlinearity is not None:
#             x = nonlinearity(x)
#         return x
#
#
#
#
# def weight_normalization(input, init_scale=1.):
#     scale_init = init_scale / tf.sqrt(v_init + 1e-10)
#     g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
#
#     V = tf.get_variable('V', [int(input.get_shape()[1]), num_units], tf.float32,
#                         tf.random_normal_initializer(0, 0.05), trainable=True)
#     V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
#     input = tf.matmul(input, V)
#     scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
#     input = tf.reshape(scaler, [1, num_units]) * input + tf.reshape(b, [1, num_units])


def l2_loss(out, y):
    with tf.variable_scope('l2_loss'):
        loss = tf.squared_difference(out, y)
        loss = tf.reduce_mean(loss)
    return loss


def l1_loss(out, y):
    with tf.variable_scope('l1_loss'):
        loss = tf.abs(out - y)
        loss = tf.reduce_mean(loss)
    return loss


def discretized_mol_loss(mu, stdv, log_pi, y, n_mix, n_classes=256, weight_const=0.):
    '''
    Selecting the right number of classes (n_classes) is important according to the range of y.
    :param out: (b, t, h)
    :param y: (b, t, 1)
    :param n_mix: 
    :return: 
    '''
    with tf.variable_scope('discretized_mol_loss'):
        y = tf.tile(y, [1, 1, n_mix])

        centered_x = y - mu
        inv_stdv = 1 / (stdv + 1e-12)
        plus_in = inv_stdv * (centered_x + 1. / n_classes)
        min_in = inv_stdv * (centered_x - 1. / n_classes)
        cdf_plus = tf.sigmoid(plus_in)
        cdf_min = tf.sigmoid(min_in)

        # log probability for edge case
        log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
        log_one_minus_cdf_min = -tf.nn.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        log_prob = tf.where(y < 0.001, log_cdf_plus,
                            tf.where(y > 0.999, log_one_minus_cdf_min, tf.log(tf.maximum(cdf_delta, 1e-12))))

        # tf.summary.histogram('prob', tf.exp(log_prob))

        log_prob = log_prob + log_pi

        tf.summary.histogram('prob_max', tf.reduce_max(tf.exp(log_prob), axis=-1))

        log_prob = tf.reduce_logsumexp(log_prob, axis=-1)

        loss_mle = -tf.reduce_mean(log_prob)

        # regularize keeping modals away from each other
        # mean = tf.reduce_sum(mu * log_pi, axis=-1, keepdims=True)
        # loss_reg = tf.reduce_sum(log_pi * tf.squared_difference(mu, mean), axis=-1)
        # loss_reg = -tf.reduce_mean(loss_reg)
        # loss = loss_mle + weight_const * loss_reg
        loss = loss_mle

        # tf.summary.scalar('loss_mle', loss_mle)
        # tf.summary.scalar('loss_mix', loss_mix)

        return loss


def power_loss(out, y, win_length, hop_length):
    def power(wav):
        stft_matrix = stft(wav, win_length, hop_length)
        return tf.square(tf.abs(stft_matrix))

    loss = tf.squared_difference(power(out), power(y))
    loss = tf.reduce_mean(loss)
    return loss


class LinearIAFLayer(object):
    def __init__(self, batch_size, scaler, shifter):
        self.batch_size = batch_size
        self.scaler = scaler
        self.shifter = shifter

    # network
    def __call__(self, input, condition=None):
        '''
        input = (n, t, h), condition = (n, t, h)
        '''
        scale = self.scaler(input, condition)
        shift = self.shifter(input, condition)
        out = input * scale + shift
        return out