# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import get_current_tower_context

from hparam import hparam as hp
from modules import LinearIAFLayer, WaveNet, normalize, power_loss, l1_loss


class IAFVocoder(ModelDesc):
    '''
    wav: [-1, 1]
    '''

    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        self.t_mel = 1 + length // hp.signal.hop_length
        self.length = length

    # network
    def __call__(self, wav, melspec, is_training, name='iaf_vocoder'):

        if hp.train.use_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.998)

        # use_ema = True if hp.train.use_ema and not get_current_tower_context().is_training else False

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  #, custom_getter=_ema_getter if use_ema else None):
            with tf.variable_scope('cond'):
                condition = self._upsample_cond(melspec, is_training=is_training, strides=[4, 4, 5])  # (n, t, h)
                if hp.model.normalize_cond:
                    with tf.variable_scope('normalize'):
                        condition = normalize(condition, method=hp.model.normalize_cond, is_training=is_training)

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
                        normalize=hp.model.normalize_wavenet,
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
                        normalize=hp.model.normalize_wavenet,
                    )
                    iaf = LinearIAFLayer(batch_size=hp.train.batch_size, scaler=scaler, shifter=shifter)
                    input = iaf(input, condition if hp.model.condition_all_iaf or i is 0 else None)  # (n, t, h)

                # normalization
                input = normalize(input, method=hp.model.normalize, is_training=is_training, name='normalize{}'.format(i))

        if hp.train.use_ema:
            var_class = tf.trainable_variables('iaf_vocoder')
            ema_op = ema.apply(var_class)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

        return input

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, self.length, 1), 'wav'),  # (n, t)
                InputDesc(tf.float32, (None, self.t_mel, hp.signal.n_mels), 'melspec')]  # (n, t_mel, n_mel)

    def _build_graph(self, inputs):
        wav, melspec = inputs
        is_training = get_current_tower_context().is_training

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
                cond = normalize(cond, method=hp.model.normalize_cond, is_training=is_training,
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