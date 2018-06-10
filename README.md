# Parallel WaveNet Vocoder

## Overview
This is a experimental project that build a WaveNet-based vocoder which converts mel-spectrogram to raw wave in parallel, inspired by [parallel WaveNet]() paper.

Thanks to the structure of the inverse autoregressive flow(IAF), one of non-autoregressive models, we're able to generate sequential data in parallel. Because to optimize the IAF model directly in maximum likelihood estimation(MLE) fails to find appropriate optimum so that generalization is hard, the paper introduces an alternative method a.k.a probability density distillation. 
It trains the original WaveNet beforehand and then optimize the IAF model to model similar output probability by minimizing KL divergence between two probabilities. If we think a bit more, we could conclude that the 'autoregressive' loss is forced to the IAF model, which is a non-autoregressive model itself. That means the 'autoregressive' constraint is still the key when it comes to training sequence generation models.

To optimize the model, I've only tried the simpler one (MLE) because I was curious about the motivation of to devise the alternative one. In conclusion, to optimize the IAF model without autoregressive constraint was almost not feasible in my case. Please refer to the output samples [here](https://soundcloud.com/andabi/sets/parallel-wavenet-vocoder).

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

## Samples
The dataset I tested on is a bunch of audio files uttered by a female speaker(slt) in CMU arctic dataset. Please refer to the output samples [here](https://soundcloud.com/andabi/sets/parallel-wavenet-vocoder).

## Discussion
* Why optimizing the IAF model harder?
** Entropy of P(x_t | z_<=t) is larger than that of P(x_t | x_<t). 
That means the modality of the probability of the dataset could be not clearer in the former case.
