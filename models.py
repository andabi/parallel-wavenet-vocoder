# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

from hparam import hparam as hp
from modules import LinearIAFLayer, WaveNet, discretized_mol_loss, l2_loss, normalize, power_loss, l1_loss
from tensorpack.tfutils import get_current_tower_context
import numpy as np

class IAFVocoder(ModelDesc):
    '''
    wav: [-1, 1]
    '''

    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        # self.t_mel = 1 + hp.signal.max_length // hp.signal.hop_length
        self.t_mel = 1 + length // hp.signal.hop_length
        self.length = length

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, self.length, 1), 'wav'),  # (n, t)
                InputDesc(tf.float32, (None, self.t_mel, hp.signal.n_mels), 'melspec')]  # (n, t_mel, n_mel)

    def _build_graph(self, inputs):
        wav, melspec = inputs
        is_training = get_current_tower_context().is_training

        if hp.train.loss == 'mol':
            with tf.variable_scope('iaf_vocoder'):
                mu, stdv, log_pi = self(*inputs, is_training=is_training)
                tf.summary.histogram('mu', mu)
                tf.summary.histogram('var', stdv)
                tf.summary.histogram('pi', tf.exp(log_pi))
            l_loss = discretized_mol_loss(mu, stdv, log_pi, y=wav, n_mix=hp.train.n_mix)
            out = self.generate(mu, log_pi)
        else:
            with tf.variable_scope('iaf_vocoder'):
                out = self(*inputs, is_training=is_training)
            out = tf.identity(out, name='pred_wav')
            l_loss = l1_loss(out=out, y=wav)

        with tf.name_scope('loss'):
            p_loss = power_loss(out=tf.squeeze(out, -1), y=tf.squeeze(wav, -1),
                                win_length=hp.signal.win_length, hop_length=hp.signal.hop_length)
            tf.summary.scalar('likelihood', l_loss)
            self.cost = l_loss + hp.train.weight_power_loss * p_loss
            if hp.train.weight_power_loss > 0:
                tf.summary.scalar('power', p_loss)
            tf.summary.scalar('total_loss', self.cost)

        # build graph for generation phase.
        if not is_training:
            tf.summary.histogram('hist/wav', wav)
            tf.summary.histogram('hist/out', out)
            tf.summary.audio('audio/pred', out, hp.signal.sr)
            tf.summary.audio('audio/gt', wav, hp.signal.sr)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr, beta2=hp.train.adam_beta2)

    def _upsample_cond(self, melspec, is_training, strides):
        assert(np.prod(np.array(strides)) == hp.signal.hop_length)

        # option1) Upsample melspec to fit to shape of waveform. (n, t_mel, n_mel) => (n, t, h)
        if hp.model.cond_upsample_method == 'transposed_conv':
            cond = tf.expand_dims(melspec, 1)
            length = self.t_mel
            input_channels = hp.signal.n_mels
            for i, stride in enumerate(strides):
                w = tf.get_variable('transposed_conv_{}_weights'.format(i),
                                     shape=(1, stride, hp.model.condition_channels, input_channels))
                input_channels = hp.model.condition_channels
                length *= stride
                cond = tf.nn.conv2d_transpose(cond, w, output_shape=(
                    self.batch_size, 1, length, hp.model.condition_channels), strides=[1, 1, stride, 1])
                cond = tf.nn.relu(cond)
                cond = normalize(cond, method=hp.model.normalize, is_training=is_training,
                                 name='normalize_transposed_conv_{}'.format(i))
            cond = tf.squeeze(cond, 1)
            cond = cond[:, hp.signal.hop_length // 2: -hp.signal.hop_length // 2, :]  # (n, t, h)

        # option2) just copy value and expand dim of time step
        elif hp.model.cond_upsample_method == 'repeat':
            cond = tf.layers.dense(melspec, units=hp.model.condition_channels, activation=tf.nn.relu)
            cond = tf.reshape(tf.tile(cond, [1, 1, hp.signal.hop_length]),
                              shape=[-1, self.t_mel * hp.signal.hop_length, hp.model.condition_channels])
            cond = cond[:, hp.signal.hop_length // 2: -hp.signal.hop_length // 2, :]
        else:
            cond = None
        return cond

    def generate(self, mu, log_pi):
        argmax = tf.one_hot(tf.argmax(log_pi, axis=-1), hp.train.n_mix)
        pred = tf.reduce_sum(mu * argmax, axis=-1, name='pred_wav', keepdims=True)
        return pred

    @auto_reuse_variable_scope
    # network
    def __call__(self, wav, melspec, is_training):
        with tf.variable_scope('cond'):
            condition = self._upsample_cond(melspec, is_training=is_training, strides=[4, 4, 5])  # (n, t, h)
            if hp.model.normalize and not hp.model.no_norm_cond:
                with tf.variable_scope('normalize'):
                    condition = normalize(condition, method=hp.model.normalize, is_training=is_training)

        # Sample from logistic dist.
        logstic_dist = tf.contrib.distributions.Logistic(loc=0., scale=1.)
        input = logstic_dist.sample([self.batch_size, self.length, 1])
        for i in range(hp.model.n_iaf):
            with tf.variable_scope('iaf{}'.format(i)):
                scaler = WaveNet(
                    batch_size=self.batch_size,
                    dilations=hp.model.dilations[i],
                    filter_width=hp.model.filter_width,
                    residual_channels=hp.model.residual_channels,
                    dilation_channels=hp.model.dilation_channels,
                    quantization_channels=1,
                    skip_channels=hp.model.skip_channels,
                    use_biases=hp.model.use_biases,
                    condition_channels=hp.model.condition_channels,
                    use_skip_connection=hp.model.use_skip_connection,
                    is_training=is_training,
                    name='scalar',
                    normalize=hp.model.normalize
                )
                shifter = WaveNet(
                    batch_size=self.batch_size,
                    dilations=hp.model.dilations[i],
                    filter_width=hp.model.filter_width,
                    residual_channels=hp.model.residual_channels,
                    dilation_channels=hp.model.dilation_channels,
                    quantization_channels=1,
                    skip_channels=hp.model.skip_channels,
                    use_biases=hp.model.use_biases,
                    condition_channels=hp.model.condition_channels,
                    use_skip_connection=hp.model.use_skip_connection,
                    is_training=is_training,
                    name='shifter',
                    normalize=hp.model.normalize,
                )
                iaf = LinearIAFLayer(batch_size=hp.train.batch_size, scaler=scaler, shifter=shifter)
                input = iaf(input, condition if hp.model.condition_all_iaf or i is 0 else None)  # (n, t, h)

            # normalization
            input = normalize(input, method=hp.model.normalize, is_training=is_training, name='normalize{}'.format(i))

        # if hp.train.loss != 'mol':
        return input

        # parameters of MoL
        # with tf.variable_scope('mol'):
        #     n_mix = hp.train.n_mix
        #     w = tf.get_variable('conv_weights', shape=(1, hp.model.quantization_channels, n_mix * 3))
        #     out = tf.nn.conv1d(input, w, stride=1, padding='SAME')  # (b, t, h) => (b, t, 3n)
        #
        #     # bias for various modals
        #     bias = tf.get_variable('bias', shape=[n_mix * 3, ],
        #                            initializer=tf.random_uniform_initializer(minval=-3., maxval=3.))
        #     out = tf.nn.bias_add(out, bias)
        #
        #     mu = out[..., :n_mix]
        #     mu = tf.nn.tanh(mu)  # (b, t, n)
        #
        #     stdv = out[..., n_mix: 2 * n_mix]
        #     stdv = tf.nn.softplus(stdv)  # (b, t, n)
        #
        #     log_pi = out[..., 2 * n_mix: 3 * n_mix]  # (b, t, n)
        #     log_pi = tf.nn.log_softmax(log_pi)
        # return mu, stdv, log_pi
