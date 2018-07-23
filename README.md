# Heartbeat arrhythmia Echo State Network - Reservoir Computing classifier
## Full test MIT single lead version (2016-2018)


Echo State Network - Reservoir Computing Project to classify heartbeat arrhythmia via single lead ECG


## Database

 Physionet/Physiobank MIT-BIH AR database used:
 
 https://www.physionet.org/physiobank/database/mitdb/ 
 
 http://academy.beecardia.com/physiobank/database/mitdb
 
 Preprocessed databased arranged in simple .txt files
 Files MIT/leadA.zip and MIT/leadB.zip need to be unzipped before use

 Target SVE class

### Preprocessing

 ECG recordings are filtered in a bandwidth [0.5,35Hz]for noise removal and baseline correction.

 Sampling rate interpolation to 250Hz.
 
 Filters: Butterworth high-pass filter (cutoff 0.5 Hz), Finite impulse response filter of 12th order (35 Hz, at 3-dB point). 
 
 Paced ECG patients removed. 
 
 
## Prerequisites

Physionet MIT-BIH AR database used:
 
https://www.physionet.org/physiobank/database/mitdb/ 
 
http://academy.beecardia.com/physiobank/database/mitdb

CPU implementation based on [Eigen library](http://eigen.tuxfamily.org/)

GPU implementation based on [NVIDIA CUDA library](https://docs.nvidia.com/cuda/) (cuBLAS & cuSolver)

https://github.com/NVIDIA/cuda-samples

##Random masks
Random masks generated via pseudo-random number generation (MATLAB, GNU Octave, GNU Scientific Library). Example files available. 

## Platform

This code has been developed and tested in a GNU/Linux operative system.

In order to test the implementation in other platforms, the following aspects and platform dependent functions must be changed and commented accordingly:
1. Makefile paths
2. config.h
3. second.cpp functions (platform dependent)
4. timestamp function to provide run_info.txt output

## Authors
Work developed at the Institute for Cross-Disciplinary Physics and Complex Systems (2016-2018)

IFISC UIB-CSIC - https://ifisc.uib-csic.es/en/
 
 - M. Alfaras 
 - M. C. Soriano
 - S. Ortín

## Acknowledgments

Authors acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research.

[NVIDIA GPU Grant Program](https://developer.nvidia.com/academic_gpu_seeding)

