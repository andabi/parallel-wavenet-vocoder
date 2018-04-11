# -*- coding: utf-8 -*-
#!/usr/bin/env python


# -*- coding: utf-8 -*-
# !/usr/bin/env python


import tensorflow as tf
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore

from data_load import DataFlow
from hparam import hparam as hp
from models import IAFVocoder
import fire
import numpy as np


def get_eval_input_names():
    return ['melspec']


def get_eval_output_names():
    return ['pred_wav']


def generate(case='default', ckpt=None, gpu=None):
    '''
    :param case: experiment case name
    :param ckpt: checkpoint to load model
    :param gpu: comma separated list of GPU(s) to use
    :return: 
    '''

    hp.set_hparam_yaml(case)

    # dataflow
    df = DataFlow(hp.generate.data_path, hp.generate.batch_size)

    # model
    model = IAFVocoder(batch_size=hp.generate.batch_size)

    # samples
    gt_wav, melspec = df().get_data().next()

    ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
    print('{} loaded.'.format(ckpt))

    # predictor
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=SaverRestore(ckpt) if ckpt else None)
    generate_audio = OfflinePredictor(pred_conf)

    # feed forward
    pred_wav, = generate_audio(melspec)

    tf.summary.audio('generate/pred', pred_wav, hp.signal.sr)
    tf.summary.audio('generate/gt', gt_wav, hp.signal.sr)

    writer = tf.summary.FileWriter(hp.logdir)
    with tf.Session() as sess:
        summ = sess.run(tf.summary.merge_all())
        writer.add_summary(summ)
    writer.close()

if __name__ == '__main__':
    fire.Fire(generate)