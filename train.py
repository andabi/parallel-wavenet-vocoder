# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import multiprocessing
import os

from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated, SimpleTrainer
from tensorpack.utils import logger
from tensorpack.input_source.input_source import QueueInput
from data_load import DataFlow
from hparam import hparam as hp
from models import IAFVocoder
import tensorflow as tf
import fire
# tf.enable_eager_execution()


def train(case='default', ckpt=None, gpu=None):
    '''
    :param case: experiment case name
    :param ckpt: checkpoint to load model
    :param gpu: comma separated list of GPU(s) to use
    :return: 
    '''
    hp.set_hparam_yaml(case)

    # model
    model = IAFVocoder(batch_size=hp.train.batch_size)

    # dataflow
    df = DataFlow(hp.train.data_path, hp.train.batch_size)

    # set logger for event and model saver
    logger.set_logger_dir(hp.logdir)

    train_conf = TrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=1000, n_thread=4)),
        callbacks=[
            ModelSaver(checkpoint_dir=hp.logdir),
            # TODO EvalCallback()
        ],
        max_epoch=hp.train.num_epochs,
        steps_per_epoch=hp.train.steps_per_epoch,
    )
    ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
    if ckpt:
        train_conf.session_init = SaverRestore(ckpt)

    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        train_conf.nr_tower = len(gpu.split(','))

    if hp.train.num_gpu <= 1:
        trainer = SimpleTrainer()
    else:
        trainer = SyncMultiGPUTrainerReplicated(gpus=hp.train.num_gpu)

    launch_train_with_config(train_conf, trainer=trainer)


if __name__ == '__main__':
    fire.Fire(train)