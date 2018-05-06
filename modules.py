# -*- coding: utf-8 -*-
# !/usr/bin/env python


import tensorflow as tf
from tensorflow.contrib.signal import stft


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


def causal_conv(value, filter_, dilation, name='causal_conv'):
    def time_to_batch(value, dilation, name=None):
        with tf.name_scope('time_to_batch'):
            shape = tf.shape(value)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

    def batch_to_time(value, dilation, name=None):
        with tf.name_scope('batch_to_time'):
            shape = tf.shape(value)
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed,
                              [tf.div(shape[0], dilation), -1, shape[2]])

    with tf.name_scope(name):
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


# Renovated from https://github.com/ibab/tensorflow-wavenet
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

        self.receptive_field = WaveNet.calculate_receptive_field(
            self.filter_width, self.dilations)
        self.variables = self._create_variables(name)

    # network
    def __call__(self, input_batch, condition_batch=None):
        with tf.variable_scope(self.name):
            '''Construct the WaveNet network.'''
            outputs = []
            with tf.name_scope('causal_layer'):
                current_layer = self._create_causal_layer(input_batch)
                # with tf.name_scope('normalize_casual'):
                if self.normalize:
                    with tf.variable_scope('wavenet/causal_layer/normalize'):
                        current_layer = normalize(current_layer, method=self.normalize, is_training=self.is_training)

            # Add all defined dilation layers.
            with tf.name_scope('dilated_stack'):
                for layer_index, dilation in enumerate(self.dilations):
                    with tf.name_scope('layer{}'.format(layer_index)):
                        output, current_layer = self._create_dilation_layer(
                            current_layer, layer_index, dilation, condition_batch)
                        if self.normalize:
                            with tf.variable_scope('wavenet/dilated_stack/layer{}/normalize'.format(layer_index)):
                                current_layer = normalize(current_layer, method=self.normalize, is_training=self.is_training)
                        outputs.append(output)

            with tf.name_scope('postprocessing'):
                # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
                # postprocess the output.
                w1 = self.variables['postprocessing']['postprocess1']
                w2 = self.variables['postprocessing']['postprocess2']
                if self.use_biases:
                    b1 = self.variables['postprocessing']['postprocess1_bias']
                    b2 = self.variables['postprocessing']['postprocess2_bias']

                # We skip connections from the outputs of each layer, adding them
                # all up here.
                total = sum(outputs) if self.use_skip_connection else outputs[-1]
                transformed1 = tf.nn.relu(total)
                conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
                if self.use_biases:
                    conv1 = tf.add(conv1, b1)
                transformed2 = tf.nn.relu(conv1)
                conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
                if self.use_biases:
                    conv2 = tf.add(conv2, b2)

        return conv2

    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        receptive_field += filter_width - 1
        return receptive_field

    # TODO remove
    def _create_variables(self, name):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope(name):
            with tf.variable_scope('wavenet'):
                with tf.variable_scope('causal_layer'):
                    layer = dict()
                    layer['filter'] = tf.get_variable(
                        'filter',
                        [self.filter_width,
                         self.quantization_channels,
                         self.residual_channels])
                    var['causal_layer'] = layer

                var['dilated_stack'] = list()
                with tf.variable_scope('dilated_stack'):
                    for i, dilation in enumerate(self.dilations):
                        with tf.variable_scope('layer{}'.format(i)):
                            current = dict()
                            current['filter'] = tf.get_variable(
                                'filter',
                                [self.filter_width,
                                 self.residual_channels,
                                 self.dilation_channels])
                            current['gate'] = tf.get_variable(
                                'gate',
                                [self.filter_width,
                                 self.residual_channels,
                                 self.dilation_channels])
                            current['dense'] = tf.get_variable(
                                'dense',
                                [1,
                                 self.dilation_channels,
                                 self.residual_channels])
                            current['skip'] = tf.get_variable(
                                'skip',
                                [1,
                                 self.dilation_channels,
                                 self.skip_channels])

                            if self.condition_channels is not None:
                                current['cond_gateweights'] = tf.get_variable(
                                    'gc_gate',
                                    [1, self.condition_channels,
                                     self.dilation_channels])
                                current['cond_filtweights'] = tf.get_variable(
                                    'gc_filter',
                                    [1, self.condition_channels,
                                     self.dilation_channels])

                            if self.use_biases:
                                current['filter_bias'] = tf.get_variable(
                                    'filter_bias',
                                    [self.dilation_channels], initializer=tf.zeros_initializer)
                                current['gate_bias'] = tf.get_variable(
                                    'gate_bias',
                                    [self.dilation_channels], initializer=tf.zeros_initializer)
                                current['dense_bias'] = tf.get_variable(
                                    'dense_bias',
                                    [self.residual_channels], initializer=tf.zeros_initializer)
                                current['skip_bias'] = tf.get_variable(
                                    'slip_bias',
                                    [self.skip_channels], initializer=tf.zeros_initializer)

                            var['dilated_stack'].append(current)

                with tf.variable_scope('postprocessing'):
                    current = dict()
                    current['postprocess1'] = tf.get_variable(
                        'postprocess1',
                        [1, self.skip_channels, self.skip_channels])
                    current['postprocess2'] = tf.get_variable(
                        'postprocess2',
                        [1, self.skip_channels, self.quantization_channels])
                    if self.use_biases:
                        current['postprocess1_bias'] = tf.get_variable(
                            'postprocess1_bias',
                            [self.skip_channels], initializer=tf.zeros_initializer)
                        current['postprocess2_bias'] = tf.get_variable(
                            'postprocess2_bias',
                            [self.quantization_channels], initializer=tf.zeros_initializer)
                    var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        weights_filter = self.variables['causal_layer']['filter']
        return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               condition_batch):
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
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if condition_batch is not None:
            weights_cond_filter = variables['cond_filtweights']
            conv_filter = conv_filter + tf.nn.conv1d(condition_batch,
                                                     weights_cond_filter,
                                                     stride=1,
                                                     padding="SAME",
                                                     name="gc_filter")
            weights_cond_gate = variables['cond_gateweights']
            conv_gate = conv_gate + tf.nn.conv1d(condition_batch,
                                                 weights_cond_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        return skip_contribution, input_batch + transformed


def normalize(input, is_training, method='bn'):
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


# def weight_normalization(input):
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