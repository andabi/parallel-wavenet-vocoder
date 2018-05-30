# Parallel WaveNet Vocoder
> Work in progress.

## Overview
This is a WaveNet-based vocoder that raw wave is generated from mel-spectrogram in parallel, inspired by [parallel WaveNet]() paper.
Thanks to the inverse autoregressive flow(IAF) structure, it is possible to generate utterances in parallel.
According to the paper, there are two types of loss for training the model: 1. train the model directly by maximum likelihood estimation(MLE) or 2. train the original WaveNet and then optimize the model to have similar output probability, by minimizing KL divergence between them. (a.k.a probability density distillation)
I've tried the former because it's simpler, but it seems it's harder to optimize the model.
TBD: result samples

## Architectures
* The main architecture consists of a few [inverse autoregressive flow(IAF)](https://arxiv.org/abs/1606.04934) layer that transform some input probability to other output probability in a inverse autoregressive way.
That means where we assume that input z=(z_1, ..., z_n) and output x=(x_1, ..., x_n) are multivariate, x_t is computed only by z_1, ..., z_t.
(This is identical to say that Jacobian matrix dx/dz is triangular.)
In short, this model ensures that output data in a current timestep is only conditioned on latent values of all previous timesteps.
  * TBD: figure
* In a timestep of a IAF layer, input that follows logistic distribution transforms to other logistic distribution by scaling and shifting.
In other words, mean and variance of the input change.
These values are computed in a inverse autoregressive way.

* As the timesteps of the mel-spectrogram are shorter than raw wave, it's necessary to expand it to have the same length as the raw wave's.
As a result, two methods are experimented: 1. use transposed convolution and 2. repeat mel-spectrum multiple times.
In my case, the latter was better in quality.

## Training
* To optimize the model, I use maximum likelihood estimation(MLE) and this is identical to using L1 loss between prediction and ground truth of raw wave.
* I kept track of exponential moving averages (EMA) of all variables and use them in generation phase.
* TBD: training graph

## Results
TBD

## Discussion
* Why optimizing the IAF model harder?
** Entropy of P(x_t | z_<=t) is larger than that of P(x_t | x_<t). 
That means the modality of the probability of the dataset could be not clearer in the former case.