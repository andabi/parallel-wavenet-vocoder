# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

from hparam import hparam as hp
from modules import IAFLayer, WaveNet, discretizsed_mol_loss


class IAFVocoder(ModelDesc):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.t_mel = 1 + hp.signal.max_length // hp.signal.hop_length

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, hp.signal.max_length, 1), 'wav'),  # (n, t)
                InputDesc(tf.float32, (None, self.t_mel, hp.signal.n_mels), 'melspec'),]  # (n, t_mel, n_mel)

    def _build_graph(self, inputs):
        wav, melspec = inputs
        out = self(*inputs)
        self.cost = self.loss(out, wav)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)

    @auto_reuse_variable_scope
    def __call__(self, wav, melspec):
        with tf.variable_scope('iaf_vocoder'):
            # option1) Upsample melspec to fit to shape of waveform. (n, t_mel, n_mel) => (n, t, h)
            # _, t_mel, _ = melspec.get_shape().as_list()
            # stride1, stride2 = 10, 8
            #
            # w1 = tf.get_variable('transposed_conv_1_weights',
            #                      shape=(stride1, 1, hp.model.quantization_channels // 2, hp.signal.n_mels))
            # condition = tf.expand_dims(melspec, 2)
            # condition = tf.nn.conv2d_transpose(condition, w1, output_shape=(
            #     -1, stride1 * self.t_mel, 1, hp.model.quantization_channels // 2), strides=[1, stride1, 1, 1])
            # w2 = tf.get_variable('transposed_conv_2_weights',
            #                      shape=(stride2, 1, hp.model.quantization_channels, hp.model.quantization_channels // 2))
            # condition = tf.nn.conv2d_transpose(condition, w2, output_shape=(
            #     -1, stride2 * self.t_mel, 1, hp.model.quantization_channels), strides=[1, stride2, 1, 1])  # (n, t, h)
            # condition = tf.squeeze(condition, 2)

            # option2) just copy value and expand dim of time step
            condition = tf.reshape(tf.tile(melspec, [1, 1, hp.signal.hop_length]), shape=[-1, self.t_mel * hp.signal.hop_length, hp.signal.n_mels])
            condition = condition[:, hp.signal.hop_length // 2: -hp.signal.hop_length // 2, :]
            # condition = tf.layers.dense(condition, units=hp.model.dilation_channels)

            # Sample from unit gaussian.
            # input = tf.tile(tf.zeros_like(wav), [1, 1, hp.model.quantization_channels])
            # input = tf.py_func(np.random.normal, [input], [tf.double])  # (n, t, h)
            # input = tf.to_float(input)
            input = tf.random_normal(shape=(self.batch_size, hp.signal.max_length, hp.model.quantization_channels))

            for _ in range(hp.model.n_iaf):
                ar_model = WaveNet(
                    batch_size=self.batch_size,
                    dilations=hp.model.dilations,
                    filter_width=hp.model.filter_width,
                    residual_channels=hp.model.residual_channels,
                    dilation_channels=hp.model.dilation_channels,
                    quantization_channels=hp.model.quantization_channels,
                    skip_channels=hp.model.skip_channels,
                    use_biases=hp.model.use_biases,
                    initial_filter_width=hp.model.initial_filter_width,
                    condition_channels=hp.signal.n_mels)
                iaf = IAFLayer(batch_size=hp.train.batch_size, n_hidden_units=hp.model.quantization_channels,
                               ar_model=ar_model)
                input = iaf(input, condition)  # (n, t, h)

            out = tf.layers.dense(input, units=1)  # (n, t, 1)
            return out

    def loss(self, out, wav):
        loss = discretizsed_mol_loss(out=out, y=wav, n_mix=hp.train.n_mix, is_training=get_current_tower_context().is_training)
        return loss