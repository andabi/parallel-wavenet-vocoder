# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fire
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from data_load import Dataset
from hparam import hparam as hp
from models import IAFVocoder


def generate(case='default', ckpt=None, gpu=None, debug=False):
    '''
    :param case: experiment case name
    :param ckpt: checkpoint to load model
    :param gpu: comma separated list of GPU(s) to use
    :return: 
    '''

    hp.set_hparam_yaml(case)

    # dataset
    dataset = Dataset(hp.generate.data_path, hp.generate.batch_size, length=hp.generate.length, is_training=False)
    print('dataset size is {}'.format(len(dataset.wav_files)))

    # model
    model = IAFVocoder(batch_size=hp.generate.batch_size, length=hp.generate.length)

    # sample
    iterator = dataset().make_one_shot_iterator()
    gt_wav_op, melspec = iterator.get_next()

    # feed forward
    pred_wav_op = model(gt_wav_op, melspec, is_training=False)

    # summaries
    tf.summary.audio('audio/pred', pred_wav_op, hp.signal.sr)
    tf.summary.audio('audio/gt', gt_wav_op, hp.signal.sr)
    # tf.summary.histogram('hist/wav', gt_wav)
    # tf.summary.histogram('hist/out', pred_wav)
    summ_op = tf.summary.merge_all()

    session_config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
    )
    with tf.Session(config=session_config) as sess:
        if debug:  # session supporting tensorboard debugging.
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:{}'.format(hp.debug_port))

        # load model
        ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
        sess.run(tf.global_variables_initializer())
        if ckpt:
            var_list = None
            if hp.train.use_ema:
                var_list = {}
                for v in tf.trainable_variables('iaf_vocoder'):
                    var_list[model.ema.average_name(v)] = v
            tf.train.Saver(var_list=var_list).restore(sess, ckpt)
            print('Successfully loaded checkpoint {}'.format(ckpt))
        else:
            print('No checkpoint found at {}.'.format(hp.logdir))

        pred_wav, gt_wav, summ = sess.run([pred_wav_op, gt_wav_op, summ_op])

    # write summaries in tensorboard
    writer = tf.summary.FileWriter(hp.logdir)
    writer.add_summary(summ)
    writer.close()

    print('Done.')

if __name__ == '__main__':
    fire.Fire(generate)
