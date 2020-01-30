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

## Use

This package is compatible with version 1.13 of Tensorflow. Arguments are specified on the command line as follows: 

```
python DNN.py --nb_layers 10 
              --nb_nodes_per_layer 100 
              --nb_train_points 750 
              --train_pointset uniform_random 
              --nb_epochs 50000 
              --batch_size 750 
              --nb_trials 20 
              --train 0 
              --make_plots 0 
              --nb_test_points 609025 
              --test_pointset CC_sparse_grid 
              --blocktype default 
              --activation relu 
              --example exp_cos 
              --optimizer Adam 
              --quiet 0 
              --input_dim 8 
              --output_dim 1 
              --MATLAB_data 1 
              --trial_num 1 
              --precision single 
              --run_ID test_exp_cos 
              --use_regularizer 0 
              --reg_lambda 1e-3 
              --error_tol 5e-7 
              --initializer normal 
              --sigma 1e-1 
              --lrn_rate_schedule exp_decay
```

We also include a shell script "run_train_test_local.sh" to simplify running ensembles of trials in testing DNN performance. To use the script, modify the variables as desired and type:

```
bash run_train_test_local.sh
```

## Examples

Difficulty of approximating a piecewise continuous function with a ReLU DNN with 15 layers and 200 nodes per layer:
![difficulty of approximating a piecewise continuous function with a ReLU DNN](https://github.com/ndexter/MLFA/blob/master/images/piecewise_function_opt_difficulties.gif)

Training a ReLU network with 2 layers and 200 nodes per layer:
![training a ReLU network with 2 layers and 200 nodes per layer](https://github.com/ndexter/MLFA/blob/master/images/relu_NN_2x200.gif)

Training a ReLU network to approximate a smooth function:
![training a relu network to approximate a smooth function](https://github.com/ndexter/MLFA/blob/master/images/smooth_function.gif)

Training a ReLU network to approximate a more oscillatory function:
![oscillatory function](https://github.com/ndexter/MLFA/blob/master/images/smooth_oscillatory_function.gif)
