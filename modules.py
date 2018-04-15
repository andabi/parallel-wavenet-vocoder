# -*- coding: utf-8 -*-
# !/usr/bin/env python


import tensorflow as tf


class IAFLayer(object):
    def __init__(self, batch_size, n_hidden_units, ar_model):
        self.batch_size = batch_size
        self.n_hidden_units = n_hidden_units
        self.ar_model = ar_model

    # network
    def __call__(self, input, condition):
        '''
        input = (n, t, h), condition = (n, t, h)
        '''
        out = self.ar_model(input, condition)
        out = tf.layers.dense(out, 2 * self.n_hidden_units)  # (n, t, 2h)
        mean = out[..., :self.n_hidden_units]  # (n, t, h)
        scale = out[..., self.n_hidden_units:]  # (n, t, h)
        out = input * scale + mean
        return out


# TODO generalize: padding valid => same, no slice
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
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])

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
                 condition_channels=None):
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

        self.receptive_field = WaveNet.calculate_receptive_field(
            self.filter_width, self.dilations)
        self.variables = self._create_variables()

    # network
    def __call__(self, input_batch, condition_batch):
        # padding at the beginning by receptive field size. length = #timesteps + receptive field size
        input_batch = tf.pad(input_batch, [[0, 0], [self.receptive_field, 0], [0, 0]])
        condition_batch = tf.pad(condition_batch, [[0, 0], [self.receptive_field, 0], [0, 0]])

        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        current_layer = self._create_causal_layer(current_layer)  # length = length - 1 = #timesteps + receptive field size - 1

        output_width = tf.shape(current_layer)[1] - self.receptive_field + 1  # length = #timesteps

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        condition_batch, output_width)
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
            total = sum(outputs)
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

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                initial_channels = self.quantization_channels
                initial_filter_width = self.filter_width
                layer['filter'] = tf.get_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
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
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               condition_batch, output_width):
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
            # cut unused time steps at the beginning.
            condition_cut = tf.shape(condition_batch)[1] - tf.shape(conv_filter)[1]
            condition_batch = tf.slice(condition_batch, [0, condition_cut, 0], [-1, -1, -1])

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
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed


def normalize(inputs,
              type="bn",
              decay=.999,
              epsilon=1e-8,
              is_training=True,
              reuse=None,
              activation_fn=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but 
        the last dimension. Or if type is `ln`, the normalization is over 
        the last dimension. Note that this is different from the native 
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py` 
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type == "bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank == 2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank == 3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   zero_debias_moving_mean=True,
                                                   fused=True,
                                                   reuse=reuse)
            # restore original shape
            if inputs_rank == 2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank == 3:
                outputs = tf.squeeze(outputs, axis=1)
        else:  # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   reuse=reuse,
                                                   fused=False)
    elif type in ("ln", "ins"):
        reduction_axis = -1 if type == "ln" else 1
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [reduction_axis], keep_dims=True)
            beta = tf.get_variable("beta", shape=params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer)
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
    else:
        outputs = inputs

    if activation_fn:
        outputs = activation_fn(outputs)

    return outputs


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


def l2_loss(out, y):
    loss = tf.squared_difference(out, y)
    loss = tf.reduce_mean(loss)
    return loss, out


def discretized_mol_loss(out, y, n_mix, n_classes=1000, weight_reg=0.):
    '''
    
    :param out: (b, t, h)
    :param y: (b, t, 1)
    :param n_mix: 
    :return: 
    '''
    _, n_timesteps, _ = y.get_shape().as_list()
    out = tf.layers.dense(out, n_mix * 3,
                          bias_initializer=tf.random_uniform_initializer(minval=-3., maxval=3.))  # (b, t, 3n)

    mu = out[..., :n_mix]
    mu = tf.nn.sigmoid(mu)  # (b, t, n)

    log_var = out[..., n_mix: 2 * n_mix]
    log_var = tf.nn.softplus(log_var)  # (b, t, n)
    # log_var = tf.maximum(log_var, -7.0)  # (b, t, n)

    log_pi = out[..., 2 * n_mix: 3 * n_mix]  # (b, t, n)
    # TODO safe softmax, better idea?
    # log_pi = normalize(0log_pi, type='ins', is_training=is_training, scope='normalize_pi')
    # log_pi = log_pi - tf.reduce_max(log_pi, axis=-1, keepdims=True)
    # m, s = tf.nn.moments(log_pi, axes=[-1], keep_dims=True)
    # log_pi = (log_pi - m) / s
    log_pi = tf.nn.log_softmax(log_pi)

    # (b, t, 1) => (b, t, n)
    y = tf.tile(y, [1, 1, n_mix])

    centered_x = y - mu
    inv_stdv = tf.exp(-log_var)
    plus_in = inv_stdv * (centered_x + 1 / n_classes)
    min_in = inv_stdv * (centered_x - 1 / n_classes)
    cdf_plus = tf.sigmoid(plus_in)
    cdf_min = tf.sigmoid(min_in)

    # log probability for edge case
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    log_prob = tf.where(y < 0.001, log_cdf_plus,
                        tf.where(y > 0.999, log_one_minus_cdf_min, tf.log(tf.maximum(cdf_delta, 1e-12))))

    # tf.summary.histogram('net2/train/prob', tf.exp(log_prob))

    log_prob = log_prob + log_pi

    tf.summary.histogram('net2/prob_max', tf.reduce_max(tf.exp(log_prob), axis=-1))

    log_prob = tf.reduce_logsumexp(log_prob, axis=-1)

    loss_mle = -tf.reduce_mean(log_prob)

    # regularize keeping modals away from each other
    mean = tf.reduce_sum(mu * log_pi, axis=-1, keepdims=True)
    loss_reg = tf.reduce_sum(log_pi * tf.squared_difference(mu, mean), axis=-1)
    loss_reg = -tf.reduce_mean(loss_reg)

    loss = loss_mle + weight_reg * loss_reg

    # tf.summary.scalar('net2/train/loss_mle', loss_mle)
    # tf.summary.scalar('net2/train/loss_mix', loss_mix)

    return loss, mu, log_var, log_pi
