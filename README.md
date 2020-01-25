# MLFA 
(M)achine (L)earning (F)unction (A)pproximation

This code implements the fully-connected Deep Neural Network (DNN) architectures considered in the paper "The gap between theory and practice in function approximation with deep neural networks" available at https://arxiv.org/abs/2001.07523

The DNN.py code accepts a variety of command line inputs, specifying everything from
the optimizer to use in solving to the width, depth, and activation function of the networks.
Various building-blocks are available for constructing the networks, including the default
activation block and ResNet blocks in resnet_v1 and resnet_v2. The DNN code accepts
MATLAB .mat format data files containing an array X of points and Y of function values, and
generates "run_data.mat" and "ensemble_data.mat" files containing information about the runs
and testing data, respectively.

