# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore

from data_load import Dataset
from hparam import hparam as hp
from models import IAFVocoder
import fire
from tensorflow.python import debug as tf_debug
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator


def get_eval_input_names():
    return ['wav', 'melspec']


def get_eval_output_names():
    return ['pred_wav', 'audio/pred', 'audio/gt', 'hist/wav', 'hist/out']


def generate(case='default', ckpt=None, gpu=None, debug=False):
    '''
    :param case: experiment case name
    :param ckpt: checkpoint to load model
    :param gpu: comma separated list of GPU(s) to use
    :return: 
    '''

    hp.set_hparam_yaml(case)

    # dataflow

    # dataset
    dataset = Dataset(hp.generate.data_path, hp.generate.batch_size, length=hp.generate.length)

    # model
    model = IAFVocoder(batch_size=hp.generate.batch_size, length=hp.generate.length)

    # samples
    iterator = dataset().make_one_shot_iterator()
    with tf.Session() as sess:
        gt_wav, melspec = sess.run(iterator.get_next())

    ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
    print('{} loaded.'.format(ckpt))

    session_config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
    )
    sess = NewSessionCreator(config=session_config)
    # session supporting tensorboard debugging.
    if debug:
        sess = SessionCreatorAdapter(sess, lambda sess: tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:{}'.format(hp.debug_port)))

    # predictor
    pred_conf = PredictConfig(
        session_creator=sess,
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=SaverRestore(ckpt) if ckpt else None)
    generate_audio = OfflinePredictor(pred_conf)

    # feed forward
    _, audio_pred, audio_gt, hist_wav, hist_out = generate_audio(gt_wav, melspec)

    # write audios in tensorboard
    writer = tf.summary.FileWriter(hp.logdir)
    writer.add_summary(audio_pred)
    writer.add_summary(audio_gt)
    writer.add_summary(hist_wav)
    writer.add_summary(hist_out)
    writer.close()

    print('Done.')


if __name__ == '__main__':
    fire.Fire(generate)
