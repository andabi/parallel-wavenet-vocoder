# Parallel WaveNet Vocoder
> Work in progress.

## Overview
This is a WaveNet-based vocoder that audio is generated from mel-spectrogram as condition in parallel inspired by [parallel WaveNet]() paper.
Thanks to the model structure, we're able to generate utterances in parallel.
According to the paper, there are two ways for training the model: 1. directly train the model with maximum likelihood estimation (MLE) or 2. train the original WaveNet and enforce the model to have similar output probability by minimizing KL divergence between them. (a.k.a probability density distillation)
I've tried the former because it's simpler, but it seems it's harder to optimize the model.

## Architectures
* The main architecture consists of a few inverse autoregressive flow(IAF) layer that transform some input probability to different output probability in a autoregressive way.
That means where we assume that input variable z=(z_1, ..., z_n) and output variable x=(x_1, ..., x_n) are multivariate, x_t is affected only by z_1, ..., z_t.
In other words, Jacobian matrix dx/dz is triangular.
In short, this model ensures that the output data in a timestep should be generated from only latent values in previous timesteps.
  * TBD: figure

* As the timesteps of mel-spectrogram is more shorter than raw wave, we need to expand it to have the same length of the raw wave.
Therefore, two methods are experimented: 1. use transposed convolution and 2. repeat mel-spectrum multiple times.
In my case, the latter was better in quality.

## Training
* I kept track of exponential moving average (EMA) of all variables and use them in generation phase.

## Results
TBD