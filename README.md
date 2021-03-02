# Variational Log-Power Spectral Tracking for Acoustic Signals
Bart van Erp, İsmail Şenöz and Bert de Vries
### Submitted to 2021 IEEE Statistical Signal Processing Workshop (SSP)
#### Paper abstract
    This paper proposes a generative hierarchical probabilistic model for acoustic signals where both the frequency decomposition and log-power spectrum appear as latent variables. In order to facilitate efficient inference, we represent the model in a factor graph that includes a probabilistic Fourier transform and a Gaussian scale model as modules. We derive novel ways of performing variational message passing-based inference in the Gaussian scale model. As a result, in this model a probabilistic representation of the log-power spectrum of an acoustic signal can be effectively inferred online. The proposed model may find applications as a front end wherever probabilistic log-power spectral features of a signal are needed. We validate the model and message passing-based inference methods by tracking the log-power spectrum of a speech signal. 

This repository contains the experiments and derivations of the paper. The experiments are two-fold. First the log-power spectrum of an acoustic signal is tracked. Secondly the proposed inference methods are compared.

