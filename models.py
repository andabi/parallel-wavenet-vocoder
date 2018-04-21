# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

from hparam import hparam as hp
from modules import IAFLayer, WaveNet, discretized_mol_loss, l2_loss, instance_normalization, power_loss
from tensorpack.tfutils import get_current_tower_context


class IAFVocoder(ModelDesc):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.t_mel = 1 + hp.signal.max_length // hp.signal.hop_length

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, hp.signal.max_length, 1), 'wav'),  # (n, t)
                InputDesc(tf.float32, (None, self.t_mel, hp.signal.n_mels), 'melspec')]  # (n, t_mel, n_mel)

    def _build_graph(self, inputs):
        wav, melspec = inputs
        is_training = get_current_tower_context().is_training

        if hp.train.loss is 'mol':
            with tf.variable_scope('iaf_vocoder'):
                mu, stdv, log_pi = self(*inputs)
            out = self.generate(mu, log_pi)
            l_loss = discretized_mol_loss(mu, stdv, log_pi, y=wav, n_mix=hp.train.n_mix)
        else:
            with tf.variable_scope('iaf_vocoder'):
                out = self(*inputs)
            w = tf.get_variable('weights', shape=(1, out.get_shape().as_list()[-1], 1))
            out = tf.nn.conv1d(out, w, stride=1, padding='SAME')  # (b, t, h) => (b, t, 1)
            out = tf.identity(out, name='pred_wav')
            l_loss = l2_loss(out=out, y=wav)

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
            tf.summary.audio('audio/pred', out, hp.signal.sr)
            tf.summary.audio('audio/gt', wav, hp.signal.sr)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)

    def _upsample_cond(self, melspec):

        # option1) Upsample melspec to fit to shape of waveform. (n, t_mel, n_mel) => (n, t, h)
        if hp.model.cond_upsample_method == 'transposed_conv':
            stride1, stride2 = 10, 8
            assert (stride1 * stride2 == hp.signal.hop_length)
            w1 = tf.get_variable('transposed_conv_1_weights',
                                 shape=(1, stride1, hp.model.condition_channels, hp.signal.n_mels))
            cond = tf.expand_dims(melspec, 1)
            cond = tf.nn.conv2d_transpose(cond, w1, output_shape=(
                self.batch_size, 1, stride1 * self.t_mel, hp.model.condition_channels), strides=[1, 1, stride1, 1])
            w2 = tf.get_variable('transposed_conv_2_weights',
                                 shape=(1, stride2, hp.model.condition_channels, hp.model.condition_channels))
            cond = tf.nn.conv2d_transpose(cond, w2, output_shape=(
                self.batch_size, 1, hp.signal.hop_length * self.t_mel, hp.model.condition_channels),
                                          strides=[1, 1, stride2, 1])  # (n, t, h)
            cond = tf.squeeze(cond, 1)
            cond = cond[:, hp.signal.hop_length // 2: -hp.signal.hop_length // 2, :]

        # option2) just copy value and expand dim of time step
        elif hp.model.cond_upsample_method == 'repeat':
            cond = tf.layers.dense(melspec, units=hp.model.condition_channels)
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
    def __call__(self, wav, melspec):
        with tf.variable_scope('cond'):
            condition = self._upsample_cond(melspec)  # (n, t, h)
            if hp.model.normalize:
                with tf.variable_scope('normalize'):
                    condition = instance_normalization(condition)

        # Sample from unit gaussian.
        input = tf.random_normal(
            shape=(self.batch_size, hp.signal.max_length, hp.model.quantization_channels))  # (n, t, h)
        for i in range(hp.model.n_iaf):
            with tf.variable_scope('iaf{}'.format(i)):
                ar_model = WaveNet(
                    batch_size=self.batch_size,
                    dilations=hp.model.dilations,
                    filter_width=hp.model.filter_width,
                    residual_channels=hp.model.residual_channels,
                    dilation_channels=hp.model.dilation_channels,
                    quantization_channels=hp.model.quantization_channels,
                    skip_channels=hp.model.skip_channels,
                    use_biases=hp.model.use_biases,
                    condition_channels=hp.model.condition_channels)
                iaf = IAFLayer(batch_size=hp.train.batch_size, n_hidden_units=hp.model.quantization_channels,
                               ar_model=ar_model)
                input = iaf(input, condition if hp.model.condition_all_iaf or i is 0 else None)  # (n, t, h)

            # normalization
            if hp.model.normalize:
                with tf.variable_scope('normalize{}'.format(i)):
                    input = instance_normalization(input)

        if hp.train.loss is not 'mol':
            return input

        # parameters of MoL
        with tf.variable_scope('mol'):
            n_mix = hp.train.n_mix
            w = tf.get_variable('conv_weights', shape=(1, hp.model.quantization_channels, n_mix * 3))
            out = tf.nn.conv1d(input, w, stride=1, padding='SAME')  # (b, t, h) => (b, t, 3n)

            # bias for various modals
            bias = tf.get_variable('bias', shape=[n_mix * 3, ], initializer=tf.random_uniform_initializer(minval=-3., maxval=3.))
            out = tf.nn.bias_add(out, bias)

            mu = out[..., :n_mix]
            mu = tf.nn.sigmoid(mu)  # (b, t, n)

            stdv = out[..., n_mix: 2 * n_mix]
            stdv = tf.nn.softplus(stdv)  # (b, t, n)

            log_pi = out[..., 2 * n_mix: 3 * n_mix]  # (b, t, n)
            log_pi = tf.nn.log_softmax(log_pi)
        return mu, stdv, log_pi