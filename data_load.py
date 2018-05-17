# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import numpy as np
import tensorflow as tf
from tensorpack.dataflow.base import RNGDataFlow

from audio import read_wav, wav2melspec_db, trim_wav, fix_length
from hparam import hparam as hp


class DataFlow(RNGDataFlow):
    def __init__(self, data_path, batch_size, length):
        dataset = Dataset(data_path, batch_size, length)
        self.iterator = dataset().make_one_shot_iterator()

    def __call__(self, n_prefetch=1000, n_thread=1):
        return self

    def get_data(self):
        yield self.iterator.get_next()


class Dataset():
    def __init__(self, data_path, batch_size, length):
        self.batch_size = batch_size
        self.wav_files = glob.glob(data_path)
        self.length = length

    def __call__(self, n_prefetch=1000, n_thread=32):
        dataset = tf.data.Dataset.from_tensor_slices(self.wav_files)
        dataset = dataset.map(
            lambda file: tf.py_func(self._get_wav_and_melspec, [file, self.length], [tf.float32, tf.float32]),
            num_parallel_calls=n_thread)
        dataset = dataset.repeat().batch(self.batch_size).prefetch(n_prefetch)
        return dataset

    @staticmethod
    def _get_wav_and_melspec(wav_file, length):
        wav = read_wav(wav_file, sr=hp.signal.sr)
        wav = trim_wav(wav)
        if length:
            wav = fix_length(wav, length)

        melspec = wav2melspec_db(wav, sr=hp.signal.sr, n_fft=hp.signal.n_fft, win_length=hp.signal.win_length,
                                 hop_length=hp.signal.hop_length, n_mels=hp.signal.n_mels,
                                 min_db=hp.signal.min_db, max_db=hp.signal.max_db)
        wav = np.expand_dims(wav, -1)
        return wav, melspec.astype(np.float32)
