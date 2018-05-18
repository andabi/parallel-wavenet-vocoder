# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorpack.callbacks.graph import RunUpdateOps
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated, SimpleTrainer
from tensorpack.utils import logger
from tensorpack.input_source.input_source import TFDatasetInput
from data_load import Dataset
from hparam import hparam as hp
from models import IAFVocoder
import tensorflow as tf
import fire
from utils import remove_all_files
from tensorpack.callbacks.saver import ModelSaver


def train(case='default', ckpt=None, gpu=None, r=False):
    '''
    :param case: experiment case name
    :param ckpt: checkpoint to load model
    :param gpu: comma separated list of GPU(s) to use
    :param r: start from the beginning.
    :return: 
    '''

    hp.set_hparam_yaml(case)
    if r:
        remove_all_files(hp.logdir)

    # model
    model = IAFVocoder(batch_size=hp.train.batch_size, length=hp.signal.max_length)

    # dataset
    dataset = Dataset(hp.train.data_path, hp.train.batch_size, length=hp.signal.max_length)
    print('dataset size is {}'.format(len(dataset.wav_files)))

    # set logger for event and model saver
    logger.set_logger_dir(hp.logdir)

    train_conf = TrainConfig(
        model=model,
        data=TFDatasetInput(dataset()),
        callbacks=[
            ModelSaver(checkpoint_dir=hp.logdir),
            RunUpdateOps()  # for batch norm, exponential moving average
            # TODO GenerateCallback()
        ],
        max_epoch=hp.train.num_epochs,
        steps_per_epoch=hp.train.steps_per_epoch,
    )
    ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
    if ckpt:
        train_conf.session_init = SaverRestore(ckpt)

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu))
        train_conf.nr_tower = len(gpu)

    if hp.train.num_gpu <= 1:
        trainer = SimpleTrainer()
    else:
        trainer = SyncMultiGPUTrainerReplicated(gpus=hp.train.num_gpu)

    launch_train_with_config(train_conf, trainer=trainer)


if __name__ == '__main__':
    fire.Fire(train)


# class GenerateCallback(Callback):
#     def _setup_graph(self):
#         self.generator = self.trainer.get_predictor(
#             get_eval_input_names(),
#             get_eval_output_names())
#         self.df = DataFlow(hp.generate.data_path, hp.generate.batch_size)
#         self.writer = tf.summary.FileWriter(hp.logdir)
#
#     def _trigger_epoch(self):
#         if self.epoch_num % hp.generate.every_n_epoch == 0:
#             gt_wav, melspec = self.df().get_data().next()
#             _, audio_pred, audio_gt = self.generator(gt_wav, melspec)
#
#             # write audios in tensorboard
#             self.writer.add_summary(audio_pred)
#             self.writer.add_summary(audio_gt)
#             self.writer.flush()
#
#     def _after_train(self):
#         self.writer.close()
