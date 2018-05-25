# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import numpy as np
import tensorflow as tf

from audio import read_wav, wav2melspec_db, trim_wav, fix_length
from hparam import hparam as hp
import math
import random


class Dataset():
    def __init__(self, data_path, batch_size, length, is_training=True):
        self.batch_size = batch_size
        wav_files = glob.glob(data_path)
        dataset_cut_idx = int(len(wav_files) * hp.train.dataset_ratio)
        self.wav_files = wav_files[:dataset_cut_idx] if is_training else wav_files[dataset_cut_idx:]
        self.length = length
        self.is_training = is_training

    def __call__(self, n_prefetch=1000, n_thread=32):
        dataset = tf.data.Dataset.from_tensor_slices(self.wav_files)
        dataset = dataset.shuffle(len(self.wav_files))
        dataset = dataset.map(
            lambda file: tf.py_func(self._get_wav_and_melspec, [file, self.length, self.is_training], [tf.float32, tf.float32]),
            num_parallel_calls=n_thread)
        dataset = dataset.repeat().batch(self.batch_size).prefetch(n_prefetch)
        return dataset

    @staticmethod
    def _get_wav_and_melspec(wav_file, length, is_training=True):
        '''
        the range of values of wav is [-1, 1].
        '''

        wav = read_wav(wav_file, sr=hp.signal.sr)
        wav = trim_wav(wav)
        # divide wav into chunks that have the given length and one is randomly selected in training, but first chunk in generation.
        n_clips = math.ceil(len(wav) / length) if is_training else 1
        idx = random.randrange(n_clips)
        start, end = length * idx, length * (idx + 1)
        wav = wav[start:end]
        assert(len(wav) <= length)
        wav = fix_length(wav, length)  # padding in case of last chunk.

        melspec = wav2melspec_db(wav, sr=hp.signal.sr, n_fft=hp.signal.n_fft, win_length=hp.signal.win_length,
                                 hop_length=hp.signal.hop_length, n_mels=hp.signal.n_mels,
                                 min_db=hp.signal.min_db, max_db=hp.signal.max_db)
        wav = np.expand_dims(wav, -1)
        return wav, melspec.astype(np.float32)
