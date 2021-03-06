# import standard libraries
import time, os, argparse, io, shutil, sys, math, socket

# import tensorflow, numpy, matplotlib, and scipy io
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import ops
import numpy as np
import hdf5storage
import matplotlib
#matplotlib.use('Agg')
import scipy.io as sio

# import plotting and data-manipulation tools
import matplotlib.pyplot as plt
import pandas as pd
import csv


# get directory
#dir = os.path.dirname(os.path.realpath(__file__))
scratchdir = '/home/ndexter/scratch'
projectdir = '/home/ndexter/projects/def-adcockb/ndexter'
#scratchdir = '/home/nick/scratch'
#projectdir = '/media/nick/Fast Secondary/scratch_backup/scratch'

# ensure matplotlib is using canonical renderer
matplotlib.get_backend()

np_d_sqrt_exp_lambdafunc = lambda x: np_d_sqrt_exp(x).astype(np.float32)

sigma = 0.1 


def get_batch(X_in, Y_in, batch_size):
    """
    Implementation of batching with random shuffling for training DNNs on 
    subsets of data

    Args:
        X_in: the points 
        Y_in: the value of the function at the points
        batch_size: the desired size of the batches 

    Returns:
        generator yielding the batches created from X_in and Y_in
    """

    # get the shapes of the input data,
    # here the first dimension is the dimensionality 
    # of the points and output function, while the second
    # is the number of samples
    X_rows = X_in.shape[0]
    Y_rows = Y_in.shape[0]

    X_cols = X_in.shape[1]
    Y_cols = Y_in.shape[1]
    
    # randomly shuffle the temporary variables along the columns
    shuffler = np.random.permutation(X_cols)
    Xtmp = X_in[:,shuffler]
    Ytmp = Y_in[:,shuffler]

    # iterate over the number of samples divided by the batch size
    for i in range(X_cols//batch_size):

        # get the index
        idx = i*batch_size

        # generator to return the batches
        yield Xtmp.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(X_rows,batch_size), \
              Ytmp.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(Y_rows,batch_size)

def default_block(x, layer, dim1, dim2, weight_bias_initializer, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of a simple rho(W*x+b) layer

    Args:
        x: a tensor
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """

    # weights for current layer
    W = tf.get_variable(name = 'l' + str(layer) + '_W', shape = [dim1, dim2],
            initializer = weight_bias_initializer, dtype = precision)
    # biases for current layer
    b = tf.get_variable(name = 'l' + str(layer) + '_b', shape = [dim2, 1], 
            initializer = weight_bias_initializer, dtype = precision)

    # return the activation of the linear map
    return rho(tf.matmul(W, x) + b)


def poly_block(x, layer, dim1, dim2, weight_bias_initializer, order = 2, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of a power of a ReLU(W*x+b) layer

    Args:
        x: a tensor
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        order: the order of the power to apply to the activation (default 2)
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """
    # weights for current layer
    W = tf.get_variable(name = 'l' + str(layer) + '_W', shape = [dim1, dim2],
            initializer = weight_bias_initializer, dtype = precision)

    # biases for current layer
    b = tf.get_variable(name = 'l' + str(layer) + '_b', shape = [dim2, 1], 
            initializer = weight_bias_initializer, dtype = precision)

    # the linear mapping
    z = tf.matmul(W, x) + b

    # return a power of the activation of the linear map
    return tf.pow(rho(z), order) 


def resnet_building_block_v1(x, layer, dim1, dim2, weight_bias_initializer, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of the ResNet v2 building block proposed in:
        https://arxiv.org/pdf/1512.03385.pdf

    Adapted from TF official ResNet model:
        https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

    Args:
        x: a tensor
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """

    # shortcut for identity mapping
    s = x 
    
    # weights for current layer
    W = tf.get_variable(name = 'l' + str(layer) + '_W', shape = [dim1, dim2],
            initializer = weight_bias_initializer, dtype = precision)

    # biases for current layer
    b = tf.get_variable(name = 'l' + str(layer) + '_b', shape = [dim2, 1], 
            initializer = weight_bias_initializer, dtype = precision)

    # activated linear mapping
    x = rho(tf.matmul(W, x) + b)

    # Identity skips previous operation
    x = x + s

    # apply a final activation to the result
    return rho(x)


def resnet_building_block_v2(x, layer, dim1, dim2, weight_bias_initializer, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of the ResNet v2 building block proposed in:
        https://arxiv.org/pdf/1603.05027.pdf

    Adapted from TF official ResNet model:
        https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

    Args:
        x: a tensor
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """

    # shortcut for identity mapping
    s = x 
    
    # weights for current layer
    W = tf.get_variable(name = 'l' + str(layer) + '_W', shape = [dim1, dim2],
            initializer = weight_bias_initializer, dtype = precision)

    # biases for current layer
    b = tf.get_variable(name = 'l' + str(layer) + '_b', shape = [dim2, 1], 
            initializer = weight_bias_initializer, dtype = precision)

    # activated linear mapping
    z = rho(tf.matmul(W, x) + b)

    # apply the skip connection after activation
    return z + s


def rankone_block(x, layer, dim1, dim2, weight_bias_initializer, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of a simple ReLU(W*x+b) layer where W is a rank one matrix formed
    by an outer product W = w*w^T for a vector w

    Args:
        x: a tensor
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """

    # weights for current layer
    w1 = tf.get_variable(name = 'l' + str(layer) + '_w1', shape = [dim1, 1],
            initializer = weight_bias_initializer, dtype = precision)

    # biases for current layer
    b = tf.get_variable(name = 'l' + str(layer) + '_b', shape = [dim2, 1], 
            initializer = weight_bias_initializer, dtype = precision)

    # take the dot product of w1 with x
    c = tf.tensordot(tf.transpose(w1),x,1)

    # return the activated result
    return rho(c*w1 + b)


def CS_block(x, W, b, layer, dim1, dim2, weight_bias_initializer, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of a stacked layer with the same weight matrix/bias 

    Args:
        x: a tensor
        W: a tensor of weights
        b: a tensor of biases
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """

    # apply the default mapping with the passed-in W and b
    return rho(tf.matmul(W, x) + b)


def CS_block_v2(x, W, layer, dim1, dim2, weight_bias_initializer, rho = tf.nn.relu, precision = tf.float64):
    """ 
    Implementation of a block re-using the weight matrix W but a new bias vector b

    Args:
        x: a tensor
        W: a tensor of weights
        b: a tensor of biases
        layer: current layer (used in naming)
        dim1: dimension of the input layer
        dim2: dimension of the output layer
        weight_bias_initializer: the initializer to use, e.g., normal, uniform, constant
        rho: the activation to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)

    Returns:
        Output tensor resulting from the block operations
    """

    ## biases for current layer
    b = tf.get_variable(name = 'l' + str(layer) + '_b', shape = [dim2, 1], 
            initializer = weight_bias_initializer, dtype = precision)

    # apply the default mapping with the passed-in W
    return rho(tf.matmul(W, x) + b)


def sqrt_exp(x):
    """
    test function
    """
    a = 1.00
    k = a
    lam = a
    #return a*np.power(x,k)*np.exp(lam*x)
    return a*tf.pow(x,k)*tf.exp(lam*x)

def soft_threshold(x):
    """
    test function
    """
    tau = 1.500
    return 2.0*(tf.nn.relu(x-tau) - tf.nn.relu(-x-tau))


def funcApprox(x, layers = 1, input_dim = 1, output_dim = 1, hidden_dim = 200, blocktype = 'default', activation = 'relu', precision = tf.float64, initializer = 'normal'):
    """
    The DNN construction method, uses the pre-defined block types to construct the net

    Args:
        x: a tensor
        layers: the number of hidden layers (default 1)
        input_dim: the dimension of the input space for the function (default 1)
        output_dim: the dimension of the output space for the function (default 1)
        hidden_dim: the number of nodes on the hidden layers (default 200)
        blocktype: the type of block to apply on the hidden layers (default 'default' given by rho(W*x+b))
        activation: the activation function to apply (default relu)
        precision: either tf.float64 (double) or tf.float32 (single)
        initializer: the initializer to use for the weights and biases (default normal)

    Output:
        A tensor representing the output of the neural network architecture
    """

    print('Constructing the tensorflow nn graph')

    # set up the weight and bias initializers from the following choices
    if initializer == 'normal':
        weight_bias_initializer = tf.random_normal_initializer(stddev = sigma, dtype = precision)
    elif initializer == 'uniform':
        weight_bias_initializer = tf.random_uniform_initializer(minval = -1.0*sigma, maxval = sigma, dtype = precision)
    elif initializer == 'constant':
        weight_bias_initializer = tf.constant_initializer(sigma, dtype = precision)
    else: 
        sys.exit('args.initializer must be one of the supported types, e.g., normal, uniform, etc.')

    # set the activation function (assume all layers use same activation
    if activation == 'relu':
        rho = tf.nn.relu
        print('Using ReLU function as activation rho')
    elif activation == 'relu6':
        rho = tf.nn.relu6
        print('Using ReLU6 function as activation rho')
    elif activation == 'crelu':
        rho = tf.nn.crelu
        print('Using CReLU function as activation rho')
    elif activation == 'elu':
        rho = tf.nn.elu
        print('Using eLU function as activation rho')
    elif activation == 'selu':
        rho = tf.nn.selu
        print('Using SeLU function as activation rho')
    elif activation == 'softplus':
        rho = tf.nn.softplus
        print('Using softplus function as activation rho')
    elif activation == 'softsign':
        rho = tf.nn.softsign
        print('Using softsign function as activation rho')
    elif activation == 'sigmoid':
        rho = tf.nn.sigmoid
        print('Using sigmoid function as activation rho')
    elif activation == 'tanh':
        rho = tf.nn.tanh
        print('Using tanh function as activation rho')
    elif activation == 'sin':
        rho = tf.math.sin
        print('Using sin function as activation rho')
    elif activation == 'soft_threshold':
        rho = soft_threshold
        print('Using soft thresholding function as activation rho')
    elif activation == 'custom':
        #np_sqrt_exp = np.vectorize(sqrt_exp)
        #np_d_sqrt_exp = np.vectorize(d_sqrt_exp)
        #if precision == tf.float64:
            #global np_d_sqrt_exp_lambdafunc = lambda x: np_d_sqrt_exp(x).astype(np.float64)
        rho = sqrt_exp
        print('Using custom function as activation rho')
    else: 
        sys.exit('args.activation must be one of the supported types, e.g., relu, sigmoid, etc.')

    # construct the network
    with tf.variable_scope('UniversalApproximator', reuse=tf.AUTO_REUSE):

        # input layer weights
        in_W = tf.get_variable(name = 'in_W', shape = [hidden_dim, input_dim],
                initializer = weight_bias_initializer, dtype = precision)

        # input layer biases
        in_b = tf.get_variable(name = 'in_b', shape = [hidden_dim, 1],
                initializer = weight_bias_initializer, dtype = precision)


        # apply the first linear mapping
        z = tf.matmul(in_W, x) + in_b

        # apply the activation to the result
        x = rho(z) 

        print('input ' + activation + ' layer: ' + str(input_dim) + 'x' + str(hidden_dim))

        # initialize the weights/biases first in the CS blocktype
        if blocktype == 'CS':
            opt1 = False
            # weights for every layer (reusing same weight matrix)
            W_CS = tf.get_variable(name = 'CS_W', shape = [hidden_dim, hidden_dim],
                    initializer = weight_bias_initializer, dtype = precision)

            # biases for every layer (reusing same bias vector)
            #b_CS = tf.get_variable(name = 'CS_b', shape = [hidden_dim, 1], 
                    #initializer = weight_bias_initializer, dtype = precision)

            for i in range(layers):
                if opt1:
                    x = CS_block(x, W_CS, b_CS, i, hidden_dim, hidden_dim, weight_bias_initializer, 
                                 precision = precision, rho = rho)
                    choice = 5
                    print('hidden ' + activation + '_' + blocktype + ' layer ' + str(i) + ': ' + 
                          str(hidden_dim) + 'x' + str(hidden_dim) + ' check opt ' + str(choice))
                else:
                    x = CS_block_v2(x, W_CS, i, hidden_dim, hidden_dim, weight_bias_initializer, 
                                    precision = precision, rho = rho)
                    choice = 6
                    print('hidden ' + activation + '_' + blocktype + ' layer ' + str(i) + ': ' +
                          str(hidden_dim) + 'x' + str(hidden_dim) + ' check opt ' + str(choice))

        # using any other block type than CS
        else:

            # iterate over the hidden layers
            for i in range(layers):

                # keeps track (for sanity) of the option applied
                choice = 0

                # apply the specified blocks to the x output of the first layer 
                if blocktype == 'default':
                    x = default_block(x, i, hidden_dim, hidden_dim, weight_bias_initializer, 
                                      precision = precision, rho = rho)
                    choice = 1
                elif blocktype == 'resnet_v1':
                    x = resnet_building_block_v1(x, i, hidden_dim, hidden_dim, weight_bias_initializer, 
                                                 precision = precision, rho = rho)
                    choice = 2
                elif blocktype == 'resnet_v2':
                    x = resnet_building_block_v2(x, i, hidden_dim, hidden_dim, weight_bias_initializer, 
                                                 precision = precision, rho = rho)
                    choice = 3
                elif blocktype == 'rankone':
                    x = rankone_block(x, i, hidden_dim, hidden_dim, weight_bias_initializer, 
                                      precision = precision, rho = rho)
                    choice = 4
                elif blocktype == 'poly':
                    x = poly_block(x, i, hidden_dim, hidden_dim, weight_bias_initializer, 2, 
                                   precision = precision, rho = rho)
                    choice = 5

                print('hidden ' + activation + '_' + blocktype + ' layer ' + str(i) + ': ' +
                      str(hidden_dim) + 'x' + str(hidden_dim) + ' check opt ' + str(choice))

        # output layer weights
        out_v = tf.get_variable(name = 'out_v', shape = [output_dim, hidden_dim],
                 initializer = weight_bias_initializer, dtype = precision)

        # output layer biases
        out_b = tf.get_variable(name = 'out_b', shape = [output_dim, 1],
                 initializer = weight_bias_initializer, dtype = precision)

        # apply the output layer and name it 'output'
        z = tf.add(tf.matmul(out_v, x, name = 'output_vx'), out_b, name = 'output')

        print('output layer: ' + str(hidden_dim) + 'x' + str(output_dim))

        # return the output tensor
        return z
    

def func_to_approx(x, exmp):
    """
    Deprecated method for directly defining functions in python to approximate
    (now we primarily use datasets to train, so function definitions are not needed)
    Only 1-dimensional functions are defined

    Args:
        x: input tensor
        exmp: which example to use (oscillatory, less_oscillatory, smooth, piecewise, ...)

    Output:
        The result of applying f(x) for the functions f defined below
    """
    if exmp == 'oscillatory': # highly oscillatory function
        y = tf.log(tf.sin(100*x) + 2) + tf.sin(10*x)
    elif exmp == 'less_oscillatory': # less oscillatory function, sparse expansion
        y = tf.log(tf.sin(10*x) + 2) + tf.sin(x)
    elif exmp == 'smooth': # simple smooth function
        y = tf.sin(x)
    elif exmp == 'piecewise': # piecewise continuous function
        condition1 = tf.less_equal(x, -0.5)
        condition2 = tf.less_equal(x,  0.0)
        condition3 = tf.less_equal(x,  0.5)
        y = tf.where(condition1, tf.square(x), tf.where(condition2, x + 5, tf.where(condition3, -x, tf.log(x) + 2)))
    else:
        sys.exit('args.example must be one of the predefined examples')

    # return the result y = f(x)
    return y

if __name__ == '__main__': 

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    print('Running tensorflow with version:')
    print(tf.__version__)

    # depending on where running, change scratch/project directories
    # TODO: change these when installed on your local machine!!!
    if socket.gethostname() == 'ubuntu-dev':
        scratchdir = '/home/nick/scratch'
        projectdir = '/home/nick/scratch'

    print(scratchdir)
    timestamp = str(int(time.time()));

    start_time = time.time()

    # parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_layers", default = 1, type = int, help = "Number of hidden layers")
    parser.add_argument("--nb_nodes_per_layer", default = 10, type = int, help = "Number of nodes per hidden layer")
    parser.add_argument("--nb_train_points", default = 1000, type = int, help = "Number of points to use in training")
    parser.add_argument("--train_pointset", default = 'uniform_random', type = str, help = "Type of points to use in training")
    parser.add_argument("--nb_test_points", default = 2000, type = int, help = "Number of points to use in testing")
    parser.add_argument("--test_pointset", default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing")
    parser.add_argument("--nb_epochs", default = 10000, type = int, help = "Number of epochs for training")
    parser.add_argument("--batch_size", default = 1000, type = int, help = "Number of training samples per batch")
    parser.add_argument("--nb_trials", default = 20, type = int, help = "Number of trials to run for averaging results")
    parser.add_argument("--train", default = 0, type = int, help = "Switch for training or testing")
    parser.add_argument("--make_plots", default = 0, type = int, help = "Switch for making plots")
    parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run")
    parser.add_argument("--blocktype", default = 'default', type = str, help = "Type of building block for hidden layers, e.g., ResNet vs. default")
    parser.add_argument("--activation", default = 'relu', type = str, help = "Type of activation function to use")
    parser.add_argument("--example", type = str, help = "Example function to approximate")
    parser.add_argument("--optimizer", default = 'SGD', type = str, help = "Optimizer to use in minimizing the loss")
    parser.add_argument("--initializer", default = 'normal', type = str, help = "Initializer to use for the weights and biases")
    parser.add_argument("--quiet", default = 1, type = int, help = "Switch for verbose output")
    parser.add_argument("--input_dim", default = 1, type = int, help = "Dimension of the input")
    parser.add_argument("--output_dim", default = 1, type = int, help = "Dimension of the output")
    parser.add_argument("--MATLAB_data", default = 1, type = int, help = "Switch for using MATLAB input data points")
    parser.add_argument("--trial_num", default = 0, type = int, help = "Number for the trial to run")
    parser.add_argument("--precision", default = 'single', type = str, help = "Switch for double vs. single precision")
    parser.add_argument("--use_regularizer", default = 0, type = int, help = "Switch for using regularizer")
    parser.add_argument("--reg_lambda", default = "1e-3", type = str, help = "Regularization parameter lambda")
    parser.add_argument("--error_tol", default = "5e-7", type = str, help = "Stopping tolerance for the solvers")
    parser.add_argument("--sigma", default = "1e-1", type = str, help = "Standard deviation for normal initializer, max and min for uniform symmetric initializer, constant for constant initializer")
    parser.add_argument("--lrn_rate_schedule", default = "exp_decay", type = str, help = "Standard deviation for normal initializer, max and min for uniform symmetric initializer, constant for constant initializer")
    args = parser.parse_args()

    print('using ' + args.optimizer + ' optimizer')

    if args.train:
        print('batching with ' + str(args.batch_size) + ' out of ' + 
              str(args.nb_train_points) + ' ' + args.train_pointset + 
              ' training points')

    # set the standard deviation for initializing the DNN weights and biases
    if args.initializer == 'normal':
        sigma = float(args.sigma)
        print('initializing (W,b) with N(0, ' + str(sigma) + '^2)')
    elif args.initializer == 'uniform':
        sigma = float(args.sigma)
        print('initializing (W,b) with U(-' + str(sigma) + ', ' + str(sigma) + ')')
    elif args.initializer == 'constant':
        sigma = float(args.sigma)
        print('initializing (W,b) as constant ' + str(sigma))

    # set the precision variable to initialize weights and biases in either double or single precision
    if args.precision == 'double':
        print('Using double precision') 
        precision = tf.float64
        error_tol = float(args.error_tol)
    elif args.precision == 'single':
        print('Using single precision')
        precision = tf.float32
        error_tol = float(args.error_tol)

    # set the unique run ID used in many places, e.g., directory names for output
    if args.run_ID is None:
        unique_run_ID = timestamp
    else:
        unique_run_ID = args.run_ID

    # set the seeds for numpy and tensorflow to ensure all initializations are the same
    np_seed = 0
    tf_seed = 0

    # set the update ratio for saving the trained model to the disk (i.e., if 
    # (current error)/(previous save error) <= update_ratio then save the model)
    update_ratio = 0.0625 
    print('Using checkpoint update ratio = ' + str(update_ratio))

    # record the trial number
    trial = args.trial_num

    # reset the default graph 
    tf.reset_default_graph()

    # unique key for naming results
    key = args.activation + '_' + args.blocktype + '_' + str(args.nb_layers) + 'x' + \
          str(args.nb_nodes_per_layer) + '_' + str(args.nb_train_points).zfill(6) + \
          '_pnts_' + str(error_tol) + '_tol_' + args.optimizer + '_opt'

    # the results and scratch directory can be individually specified (however for now they are the same)
    result_folder = scratchdir + '/results/' + unique_run_ID + '/' + key + '/trial_' + str(trial)
    scratch_folder = scratchdir + '/results/' + unique_run_ID + '/' + key + '/trial_' + str(trial)

    # create the result folder if it doesn't exist yet
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create the scratch folder if it doesn't exist yet
    if not os.path.exists(scratch_folder):
        os.makedirs(scratch_folder)

    # variable holding percentage of testing points with error above 10^{-k} for various thresholds k
    percs = [];

    # y true on the testing points
    y_true_test = [];

    # y true on the training points
    y_true_train_trials = [];

    # record the absolute maximum of the weights and biases for plotting later
    DNN_max_weights_trials = np.zeros(args.nb_trials)

    # initial error in determining when to checkpoint
    last_ckpt_loss = 1e16;

    # loading the data from MATLAB
    if args.MATLAB_data:

        # the training data is in MATLAB files with names in the form: 
        # (example)_func_(input_dim)_dim_(number of training points)_(type of pointset)_pts.mat
        training_data_filename = 'training_data/' + args.example + '_func_' \
                    + str(args.input_dim).zfill(4) + '_dim_' \
                    + str(args.nb_train_points).zfill(10) + '_' \
                    + args.train_pointset + '_pts.mat'

        print('Loading training data from: ' + training_data_filename)

        # load the MATLAB -v7.3 hdf5-based format mat file
        training_data = hdf5storage.loadmat(training_data_filename)

        # depending on the training model (either using deterministic or random points) 
        # set up the seeds based on the trial number or just use 0 seed in the case of 
        # random points
        if args.train_pointset == 'linspace':
            print('Using uniformly spaced points from MATLAB')

            # every trial uses same points (trials based on seed of np & tf)
            x_in_train = training_data['X']
            y_true_train = training_data['Y']

            # for deterministic data, there is only one data set 
            # (don't need to split based on trial number as below)
            x_in_train_trials = x_in_train

            # Since there are no different trial datasets, use the trial numbers to
            # initialize np and tf random seeds 
            print('Using trial_num = ' + str(trial) + ' as seed for tensorflow')
            print('Using trial_num = ' + str(trial) + ' as seed for numpy')
            np.random.seed(trial)
            tf.set_random_seed(trial)

        if args.train_pointset == 'CC_sparse_grid':
            print('Using uniformly spaced points from MATLAB')

            # every trial uses same points (trials based on seed of np & tf)
            x_in_train = training_data['X']
            y_true_train = training_data['Y']

            # for deterministic data, there is only one data set 
            # (don't need to split based on trial number as below)
            x_in_train_trials = x_in_train

            # record the quadrature weights if using SG-quadrature-regularized training
            quadrature_weights_train = training_data['W']

            # initialize np and tf random seeds to 0
            print('Using trial_num = ' + str(trial) + ' as seed for tensorflow')
            print('Using trial_num = ' + str(trial) + ' as seed for numpy')
            np.random.seed(trial)
            tf.set_random_seed(trial)

        elif args.train_pointset == 'uniform_random':
            # every trial uses same np & tf seed (trials based on sets of random points)
            print('Using uniform random points (trial ' + str(trial) + ') from MATLAB')

            # take the point data for this trial
            x_in_train_trials = training_data['X']
            x_in_train = np.transpose(x_in_train_trials[:,:,trial])

            # take the function data for this trial
            y_true_train_trials = training_data['Y']
            y_true_train = np.transpose(y_true_train_trials[:,:,trial])

            # these are both initialized above (TODO: change this to specify the seeds at command line)
            print('Using ' + str(tf_seed) + ' as seed for tensorflow')
            print('Using ' + str(np_seed) + ' as seed for numpy')
            np.random.seed(np_seed)
            tf.set_random_seed(tf_seed)

        # testing data filename has same structure as training data filename:
        # (example)_func_(input_dim)_dim_(number of training points)_(type of pointset)_pts.mat
        testing_data_filename = 'testing_data/' + args.example + '_func_' \
                    + str(args.input_dim).zfill(4) + '_dim_' \
                    + str(args.nb_test_points).zfill(10) + '_' \
                    + args.test_pointset + '_pts.mat'

        print('Loading testing data from: ' + testing_data_filename)
        testing_data = hdf5storage.loadmat(testing_data_filename)

        print('Using ' + args.test_pointset + ' testing points from MATLAB')

        # set the testing data (often smaller than the final testing data, for 
        # outputting the errors while training)
        x_in_test = np.transpose(testing_data['X'])
        y_true_test = np.transpose(testing_data['Y'])

        # quadrature weights used in reporting the error while training
        quadrature_weights_test = np.transpose(testing_data['W'])

    else:
        # if we're not using MATLAB for inputting the training/testing data, then
        # generate the equispaced points directly in python and evaluate the target
        # function at these points (TODO: remove this as no longer needed)
        print('Using linearly spaced points')
        if args.input_dim == 1:
            x_in_train_trials = np.zeros((args.nb_train_points, args.input_dim, args.nb_trials))
            x_in_train = np.linspace(-1.0, 1.0, num = args.nb_train_points).reshape(args.nb_train_points,1)
            x_in_test = np.linspace(-1.0, 1.0, num = args.nb_test_points).reshape(args.nb_test_points,1)
            for t in range(args.nb_trials):
                x_in_train_trials[:,:,t] = x_in_train.reshape(args.nb_train_points,args.input_dim)

        else:
            # code doesn't handle higher dimensional generation of data since mostly
            # rely on MATLAB
            sys.exit('Must use MATLAB data for args.input_dim > 1')
            
        # initialize np and tf random seeds to trial since data is deterministic
        print('Using ' + str(trial) + ' as seed for tensorflow')
        print('Using ' + str(trial) + ' as seed for numpy')
        np.random.seed(trial)
        tf.set_random_seed(trial)


    # TRAIN: if doing training
    if args.train:
        if not args.quiet:
            print('Running problem (key): ' + str(key))
            print('Saving to (result_folder): ' + str(result_folder))
            print('Starting trial: ' + str(trial))

        # set up the learning rate schedule from either exp_decay, linear, or constant 
        if args.lrn_rate_schedule == 'exp_decay':
            # need to specify the initial learning rate
            init_rate = 1e-3
            lrn_rate = init_rate

            # update frequency specifies how many epochs until the learning rate is 
            # decayed by a specific amount 
            update_freq = 1e3 #*args.batch_size/args.nb_train_points

            # calculate the base so that the learning rate schedule with 
            # exponential decay follows (init_rate)*(base)^(current_epoch/update_freq)
            base = np.exp(update_freq/args.nb_epochs*(np.log(error_tol)-np.log(init_rate)))

            # based on the above, the final learning rate is (init_rate)*(base)^(total_epochs/update_freq)
            print('based on init_rate = ' + str(init_rate)
                  + ', update_freq = ' + str(update_freq)
                  + ', calculated base = ' + str(base) 
                  + ', so that after ' + str(args.nb_epochs)
                  + ' epochs, we have final learning rate = '
                  + str(init_rate*base**(args.nb_epochs/update_freq)))

        elif args.lrn_rate_schedule == 'linear':
            # only need to specify the init rate for linear
            init_rate = 1e-3
            print('using a linear learning rate schedule')

        elif args.lrn_rate_schedule == 'constant':
            # only need to specify the init rate for constant (stays the same)
            init_rate = 1e-3
            print('using a constant learning rate')

        # plotting 
        # TODO: broken currently, need to fix
        if args.make_plots:
            plt.clf()
            plt.figure(1)
            plt.title('points and function values')
            plt.scatter(x_in_train, y_true_train)
            plt.draw()
            plt.show()
            plt.show(block=False) # show the plot
            print('attempted to plot training data')
        
        # build the graph for the DNN
        with tf.variable_scope('Graph') as scope:
        
            # inputs to the NN
            x = tf.placeholder(precision, shape = [args.input_dim, None], name = 'input')

            if not args.MATLAB_data and args.input_dim == 1:
                # directly define the function to approximate 
                y_true = func_to_approx(x, args.input_dim, args.output_dim, args.example)

            else:
                # define a placeholder to feed data into while training instead
                y_true = tf.placeholder(precision, shape = [args.output_dim, None], name = 'y_true')

            # construct the network using this function
            y = funcApprox(x, args.nb_layers, args.input_dim, args.output_dim, args.nb_nodes_per_layer, 
                           args.blocktype, args.activation, precision, args.initializer)

            print(y)

            # loss function
            with tf.variable_scope('Loss'):

                # ERM functional
                if args.use_regularizer:
                    if args.precision == 'double':
                        l2_reg_lambda = np.float64(args.reg_lambda)
                    elif args.precision == 'single':
                        l2_reg_lambda = np.float32(args.reg_lambda)

                    print('Using tf.losses.mean_squared_error + l2 regularization with lambda = ' + str(l2_reg_lambda))
                    vars = tf.trainable_variables()

                    #L2 Loss plus regularization term on the trainable weights & biases
                    loss = tf.reduce_mean(tf.abs(tf.square(y-y_true))) + \
                           tf.add_n([ tf.nn.l2_loss(v) for v in vars ])*l2_reg_lambda

                    #L1 Loss plus regularization term on the trainable weights & biases
                    #loss = tf.reduce_mean(tf.abs(tf.square(y-y_true))) + \
                            #tf.add_n([ tf.reduce_sum(tf.math.abs(v)) for v in vars ])*l2_reg_lambda

                    #print(loss) 

                else:

                    # the default mean squared error
                    print('Using tf.losses.mean_squared_error')
                    loss = tf.losses.mean_squared_error(y, y_true)

                loss_summary_t = tf.summary.scalar('loss', loss) 

            # use a manual learning rate update strategy
            # using a placeholder, we can pass the learning rate in at each iteration 
            learning_rate = tf.placeholder(precision, [], name='learning_rate')

            # set the optimizer to use 
            if args.optimizer == 'SGD':
                opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
                if not args.quiet:
                    print('using SGD optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'Adam':
                opt = tf.train.AdamOptimizer(learning_rate = learning_rate) 
                if not args.quiet:
                    print('using Adam optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'Adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate = learning_rate) 
                if not args.quiet:
                    print('using Adagrad optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'AdamW2':
                opt = tf.contrib.opt.AdamWOptimizer(1e-2, learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdamW optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'AdamW3':
                opt = tf.contrib.opt.AdamWOptimizer(1e-3, learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdamW optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'AdamW4':
                opt = tf.contrib.opt.AdamWOptimizer(1e-4, learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdamW optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'AdamW5':
                opt = tf.contrib.opt.AdamWOptimizer(1e-5, learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdamW optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'AdamW6':
                opt = tf.contrib.opt.AdamWOptimizer(1e-6, learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdamW optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'AdamW7':
                opt = tf.contrib.opt.AdamWOptimizer(1e-7, learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdamW optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'AdaMax':
                opt = tf.contrib.opt.AdaMaxOptimizer(learning_rate = learning_rate) 
                if not args.quiet:
                    print('using AdaMax optimizer with exponentially decaying learning rate and weight decay')
            elif args.optimizer == 'ProximalGradientDescent':
                opt = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate)
                if not args.quiet:
                    print('using ProximalGradientDescent optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'PGD_custom':
                opt = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate, 
                                                                l1_regularization_strength=1e-3, 
                                                                l2_regularization_strength=0.0)
                if not args.quiet:
                    print('using ProximalGradientDescent optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'PGD_custom2':
                opt = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate, 
                                                                l1_regularization_strength=0.0, 
                                                                l2_regularization_strength=1e-3)
                if not args.quiet:
                    print('using ProximalGradientDescent optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'ProximalAdagrad':
                opt = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate)
                if not args.quiet:
                    print('using ProximalAdagrad optimizer with exponentially decaying learning rate')
            elif args.optimizer == 'RMSProp':
                opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
                if not args.quiet:
                    print('using RMSProp optimizer with exponentially decaying learning rate')
            else:
                sys.exit('args.optimizer must be one of the preset optimizers: SGD, Adam, etc.')

            train_op = opt.minimize(loss)
            
        with tf.variable_scope('TensorboardMatplotlibInput') as scope:
            # matplotlib will give us the image as a string
            img_strbuf_plh = tf.placeholder(tf.string, shape = []) 
            # in png format
            my_img = tf.image.decode_png(img_strbuf_plh, 4) 
            # we transform that into an image summary
            img_summary = tf.summary.image('matplotlib_graph', tf.expand_dims(my_img, 0)) 


        if args.make_plots:
            plt.figure(figsize = (20, 10))
            #plt.ion()

        # array of iteration updates
        num = np.array([])
        num_perc = np.array([])

        # arrays of losses and learning rates used at iteration updates
        losses = np.array([])
        lrn_rates = np.array([])

        # keep track of the minimum loss achieved while training
        min_loss = 10;

        # keep track of the loss at the last learning rate update
        last_lrn_update_loss = 10;

        # print out the number of parameters to be trained
        tf_trainable_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        if not args.quiet:
            print('This model has ' + str(tf_trainable_vars) + ' trainable parameters')

        #print('Printing them now:')
        #for v in tf.trainable_variables():
            #print(v)

        # create the session to start initializing and training the DNN
        with tf.Session() as sess:

            # create a SummaryWriter to save data for TensorBoard
            sw = tf.summary.FileWriter(scratch_folder, sess.graph)

            # We create a Saver as we want to save our UA after training
            saver = tf.train.Saver()

            if not args.quiet:
                print('Training the NN')

            # initialize the variables with the type and precision above
            sess.run(tf.global_variables_initializer())

            # loss at last update
            last_loss = 1.0;

            for i in range(args.nb_epochs):#//args.batch_size):

                sys.stdout.flush()

                #lrn_rate = init_rate
                if args.lrn_rate_schedule == 'exp_decay':
                    lrn_rate = init_rate*(base)**(i/update_freq)
                elif args.lrn_rate_schedule == 'linear':
                    lrn_rate = init_rate*(1.0 - i/args.nb_epochs) + error_tol*(i/args.nb_epochs)
                elif args.lrn_rate_schedule == 'constant':
                    lrn_rate = init_rate

                count = 0

                # if doing batching, split the training data into batches and execute
                # one epoch of training with the given optimizer on the batches
                for x_in_train_batch, y_true_train_batch in get_batch(x_in_train, y_true_train, args.batch_size):

                    count = count + 1

                    # run the optimization process and save loss information
                    current_loss, loss_summary, current_learning_rate, _ = \
                            sess.run([loss, loss_summary_t, learning_rate, train_op],
                                     feed_dict = {x: x_in_train_batch, \
                                                  y_true: y_true_train_batch, \
                                                  learning_rate: lrn_rate})

                #print('batches over after ' + str(count) + ' batches')

                # add the current loss summary to the SummaryWriter for TensorBoard
                sw.add_summary(loss_summary, i + 1)

                # when the error has decreased enough test model and save
                if (current_loss/min_loss < update_ratio):

                    # remove the last checkpoint if it exists
                    if (os.path.exists(scratch_folder + '/chkpt')):
                        shutil.rmtree(scratch_folder + '/chkpt')

                    inputs_dict = {"input": x}
                    outputs_dict = {"output": y}
                    
                    # TODO: this method has been deprecated and should be updated
                    tf.saved_model.simple_save(sess, scratch_folder + '/chkpt', inputs_dict, outputs_dict)

                    # update the last checkpoint loss to determine whether to 
                    # keep this save or keep the end result
                    last_ckpt_loss = current_loss

                    # if the ratio current_loss/min_loss < update_ratio, then it's 
                    # also less than 1, implying current_loss <= min_loss (so update)
                    min_loss = current_loss;

                    # evaluate the model at the testing points
                    y_true_res, y_res = sess.run([y_true, y], feed_dict = {x: x_in_test, y_true: y_true_test})

                    # compute the absolute difference between the trained model 
                    # and the true data
                    absdiff = abs(y_true_test - y_res);

                    # compute percentages of points with errors above thresholds
                    perc_e1 = (1.0*(absdiff > 10)).sum()/args.nb_test_points*100.0;
                    perc_e0 = (1.0*(absdiff > 1)).sum()/args.nb_test_points*100.0;
                    perc_em1 = (1.0*(absdiff > 1e-1)).sum()/args.nb_test_points*100.0;
                    perc_em2 = (1.0*(absdiff > 1e-2)).sum()/args.nb_test_points*100.0;
                    perc_em3 = (1.0*(absdiff > 1e-3)).sum()/args.nb_test_points*100.0;
                    perc_em4 = (1.0*(absdiff > 1e-4)).sum()/args.nb_test_points*100.0;
                    perc_em5 = (1.0*(absdiff > 1e-5)).sum()/args.nb_test_points*100.0;
                    perc_em6 = (1.0*(absdiff > 1e-6)).sum()/args.nb_test_points*100.0;

                    # the l-infinity error over the test points
                    linferr = np.amax(absdiff);

                    # the l2err over the test points
                    l2err = np.sqrt(sum(np.square(absdiff)))

                    # the L2 error with respect to the uniform measure computed with 
                    # the sparse grid quadrature rule above
                    L2err = np.sqrt(abs(np.sum(np.square(absdiff)*quadrature_weights_test*2.0**(-1.0*args.input_dim))))

                    if not args.quiet:
                        print("error_cond: %s, loss: %8.4e, error: linf = %6.4f, L2 = %6.4f, %%>1ek: " \
                            "0 = %6.4f, -1 = %6.4f, -2 = %6.4f, -3 = %6.4f, -4 = %6.4f, -5 = %6.4f, -6 = %6.4f" \
                            "  lrn_rate = %6.4e"
                            % (str(i).zfill(8), current_loss, linferr, L2err, perc_e0, perc_em1, perc_em2, perc_em3, 
                                perc_em4, perc_em5, perc_em6, current_learning_rate));
                        

                # keep track of iterations, losses, and used learning rates
                num = np.append(num, i)
                losses = np.append(losses, [current_loss])
                lrn_rates = np.append(lrn_rates, [current_learning_rate])


                # if the model has converge or run out of epochs of training, or if 1000 epochs have passed
                if (i == 0) or (i % 1000 == 0) or (current_loss <= error_tol) or (i == args.nb_epochs - 1):

                    # run the model on the test inputs
                    y_true_res, y_res = sess.run([y_true, y], feed_dict = {x: x_in_test, y_true: y_true_test})

                    # update the losses
                    curnt_loss = np.array2string(current_loss, precision = 12)

                    # compute error statistics
                    absdiff = abs(y_true_test - y_res)

                    # compute the percentages above
                    perc_e1 = (1.0*(absdiff > 10)).sum()/args.nb_test_points*100.0;
                    perc_e0 = (1.0*(absdiff > 1)).sum()/args.nb_test_points*100.0;
                    perc_em1 = (1.0*(absdiff > 1e-1)).sum()/args.nb_test_points*100.0;
                    perc_em2 = (1.0*(absdiff > 1e-2)).sum()/args.nb_test_points*100.0;
                    perc_em3 = (1.0*(absdiff > 1e-3)).sum()/args.nb_test_points*100.0;
                    perc_em4 = (1.0*(absdiff > 1e-4)).sum()/args.nb_test_points*100.0;
                    perc_em5 = (1.0*(absdiff > 1e-5)).sum()/args.nb_test_points*100.0;
                    perc_em6 = (1.0*(absdiff > 1e-6)).sum()/args.nb_test_points*100.0;

                    # the l-infinity error over the training points
                    linferr = np.amax(absdiff);

                    # the l2 error over the training points
                    l2err = np.sqrt(sum(np.square(absdiff)))

                    # the L2 error with respect to the uniform measure computed with 
                    # the sparse grid quadrature rule above
                    L2err = np.sqrt(abs(np.sum(np.square(absdiff)*quadrature_weights_test*2.0**(-1.0*args.input_dim))))

                    # assign to running stats
                    if (i == 0):
                        num_perc = np.append(num_perc, i)
                        percs = [perc_e1, perc_e0, 
                                perc_em1, perc_em2, 
                                perc_em3, perc_em4, 
                                perc_em5, perc_em6];
                    else:
                        num_perc = np.append(num_perc, i)
                        percs = np.vstack([percs, [perc_e1, perc_e0, 
                                                perc_em1, perc_em2, 
                                                perc_em3, perc_em4, 
                                                perc_em5, perc_em6]]);

                    # save relevant run data for loading into MATLAB
                    run_data = {}
                    run_data['optimizer'] = args.optimizer
                    run_data['iterations'] = num
                    run_data['loss_per_iteration'] = losses
                    run_data['lrn_rates'] = lrn_rates
                    run_data['lrn_rate_schedule'] = args.lrn_rate_schedule

                    # these are only set for exp_decay
                    if args.lrn_rate_schedule == 'exp_decay':
                        run_data['base'] = base
                        run_data['update_freq'] = update_freq

                    run_data['init_rate'] = init_rate
                    run_data['error_tol'] = error_tol
                    run_data['percentiles_at_save'] = percs
                    run_data['percentiles_save_iters'] = num_perc
                    run_data['activation'] = args.activation
                    run_data['nb_layers'] = args.nb_layers
                    run_data['nb_nodes_per_layer'] = args.nb_nodes_per_layer
                    run_data['nb_train_points'] = args.nb_train_points
                    run_data['nb_epochs'] = args.nb_epochs
                    run_data['nb_trials'] = args.nb_trials
                    run_data['trial'] = trial
                    run_data['run_ID'] = unique_run_ID
                    run_data['nb_test_points'] = args.nb_test_points
                    run_data['blocktype'] = args.blocktype
                    run_data['initializer'] = args.initializer
                    run_data['example'] = args.example
                    run_data['timestamp'] = timestamp
                    run_data['run_time'] = time.time() - start_time
                    run_data['x_in_train'] = x_in_train
                    run_data['np_seed'] = np_seed
                    run_data['tf_seed'] = tf_seed
                    run_data['tf_version'] = tf.__version__
                    run_data['result_folder'] = result_folder
                    run_data['key'] = key
                    run_data['y_true_test'] = y_true_res
                    run_data['y_DNN_test'] = y_res
                    run_data['x_in_test'] = x_in_test
                    run_data['update_ratio'] = update_ratio
                    run_data['tf_trainable_vars'] = tf_trainable_vars
                    run_data['sigma'] = sigma

                    # save the resulting mat file with scipy.io
                    sio.savemat(result_folder + '/run_data.mat', run_data)

                    # plot error of approximation
                    if (args.input_dim == 1 and args.make_plots):
                        plt.subplot(231)
                        plt.title('true vs. approximation')
                        plt.plot(x_in_test, y_true_res, 'r')
                        plt.plot(x_in_test, y_res, 'g--')

                        #plt.ylim(-1,6)

                        # plot approximation
                        plt.subplot(232)
                        plt.title('approximation')
                        plt.plot(x_in_test, y_res)

                        # plot absolute error on log scale
                        plt.subplot(233)
                        plt.title('abs error')
                        plt.semilogy(x_in_test, abs(y_true_res - y_res))
                        #bottom = min(abs(y_true_res - y_res))
                        plt.ylim(1E-8,1E1)

                        # plot current loss vs. batch number
                        plt.subplot(234)
                        plt.title('loss vs. batch #')
                        plt.semilogy(num, losses)

                        #plot current loss vs. batch number on x-log scale
                        plt.subplot(235)
                        plt.title('loss vs. (log scale) batch #')
                        plt.loglog(num, losses)

                        # plot absolute error percentiles
                        plt.subplot(236)
                        if (i > 0):
                            plt.title('absolute error percentiles')
                            plt.semilogy(num_perc, percs[:,0],label = '% > 1e1')
                            plt.semilogy(num_perc, percs[:,1],label = '% > 1e0')
                            plt.semilogy(num_perc, percs[:,2],label = '% > 1e-1')
                            plt.semilogy(num_perc, percs[:,3],label = '% > 1e-2')
                            plt.semilogy(num_perc, percs[:,4],label = '% > 1e-3')
                            plt.semilogy(num_perc, percs[:,5],label = '% > 1e-4')
                            plt.semilogy(num_perc, percs[:,6],label = '% > 1e-5')
                            plt.semilogy(num_perc, percs[:,7],label = '% > 1e-6')
                            plt.legend(loc=1)

                        plt.tight_layout()
                        plt.draw() # update after all subplots are done
                        plt.show()
                        plt.show(block=False) # show the plot

                    if not args.quiet:
                        print('batch: ' + str(i).zfill(8) + ', loss: %8.4e, lrn_rate: %4.4e, seconds: %8.2f ' \
                            % (current_loss, current_learning_rate, time.time() - start_time))

                    # if we haven't converged, save the figure if plotting
                    if (current_loss > error_tol):
                        if args.make_plots:
                            plt.savefig('frames/trial_' + str(trial) + '_' + str(args.nb_layers) + \
                                        '_layer_epochs_' + str(args.nb_epochs) + '_points_' + \
                                        str(args.nb_train_points).zfill(6) + '_iter_' + str(i).zfill(8) + '.png')

                    # if we've converged to the error tolerance in the loss, or run 
                    # into the maximum number of epochs, stop training and save
                    if (current_loss <= error_tol) or (i == args.nb_epochs - 1):
                        # for saving images of outputs of the trained DNNs 
                        if not os.path.exists(scratchdir + '/results/' + unique_run_ID + '/' + key + '/trained_imgs/'):
                            os.makedirs(scratchdir + '/results/' + unique_run_ID + '/' + key + '/trained_imgs/')

                        # save the plot as a png
                        if args.make_plots:
                            plt.savefig(scratchdir + '/results/' + unique_run_ID + '/' + key + '/trained_imgs/'  \
                                    + key + '_trial_' + str(trial) + '.png')

                        # output the final checkpoint loss and statistics 
                        if not args.quiet:
                            print("final chkpt: %s, loss: %8.4e, error: linf = %6.4f, L2 = %6.4f, %%>1ek: " \
                                "0 = %6.4f, -1 = %6.4f, -2 = %6.4f, -3 = %6.4f, -4 = %6.4f, -5 = %6.4f, -6 = %6.4f" \
                                "  lrn_rate = %6.4e"
                                % (str(i).zfill(8), current_loss, linferr, L2err, perc_e0, perc_em1, perc_em2, perc_em3, 
                                    perc_em4, perc_em5, perc_em6, current_learning_rate));

                        # TODO: this save method has been deprecated and needs to be updated
                        inputs_dict = {"input": x}
                        outputs_dict = {"output": y}
                        tf.saved_model.simple_save(sess, result_folder + '/final', inputs_dict, outputs_dict)

                        # save relevant run data for loading into MATLAB
                        run_data = {}
                        run_data['optimizer'] = args.optimizer
                        run_data['iterations'] = num
                        run_data['loss_per_iteration'] = losses
                        run_data['lrn_rates'] = lrn_rates
                        run_data['lrn_rate_schedule'] = args.lrn_rate_schedule

                        # only exp_decay has base and update_freq parameters
                        if args.lrn_rate_schedule == 'exp_decay':
                            run_data['base'] = base
                            run_data['update_freq'] = update_freq

                        run_data['init_rate'] = init_rate
                        run_data['error_tol'] = error_tol
                        run_data['percentiles_at_save'] = percs
                        run_data['percentiles_save_iters'] = num_perc
                        run_data['activation'] = args.activation
                        run_data['nb_layers'] = args.nb_layers
                        run_data['nb_nodes_per_layer'] = args.nb_nodes_per_layer
                        run_data['nb_train_points'] = args.nb_train_points
                        run_data['nb_epochs'] = args.nb_epochs
                        run_data['nb_trials'] = args.nb_trials
                        run_data['trial'] = trial
                        run_data['run_ID'] = unique_run_ID
                        run_data['nb_test_points'] = args.nb_test_points
                        run_data['blocktype'] = args.blocktype
                        run_data['initializer'] = args.initializer
                        run_data['example'] = args.example
                        run_data['timestamp'] = timestamp
                        run_data['run_time'] = time.time() - start_time
                        run_data['x_in_train'] = x_in_train
                        run_data['np_seed'] = np_seed
                        run_data['tf_seed'] = tf_seed
                        run_data['tf_version'] = tf.__version__
                        run_data['result_folder'] = result_folder
                        run_data['key'] = key
                        run_data['y_true_test'] = y_true_res
                        run_data['y_DNN_test'] = y_res
                        run_data['x_in_test'] = x_in_test
                        run_data['tf_trainable_vars'] = tf_trainable_vars
                        run_data['precision'] = args.precision
                        run_data['sigma'] = sigma

                        # save the data with scipy.io as MATLAB -v7.3 hdf5-based format
                        sio.savemat(result_folder + '/run_data.mat', run_data)

                        if not args.quiet:
                            print('last chkpt loss: %8.4e  current loss: %8.4e' % (last_ckpt_loss, current_loss))

                        # if the last checkpoint actually had better error, delete the final 
                        # and replace it with the better checkpoint
                        # NOTE: since the run_data is saved above, this corresponds to the 
                        # data for the final DNN (before replacing with any checkpoints)
                        # Testing should be performed only on the final result (whether an
                        # earlier checkpoint or the DNN at this stage)
                        if last_ckpt_loss < current_loss:
                            # remove the final if it exists
                            if (os.path.exists(result_folder + '/final')):
                                shutil.rmtree(result_folder + '/final')
                            else:
                                print('folder ' + result_folder + '/final' + ' does not exist, skipping')

                            # replace the final with the checkpoint 
                            if (os.path.exists(scratch_folder + '/chkpt')):
                                shutil.move(scratch_folder + '/chkpt', result_folder + '/final')
                            else:
                                print('folder ' + scratch_folder + '/chkpt' + ' does not exist, skipping')

                            if not args.quiet:
                                print('tried replacing final with best checkpoint')

                        else:
                            # in this case the final is the best out of the 
                            # checkpoints, so remove the last checkpoint
                            if not args.quiet:
                                print('keeping final as best checkpoint')
                                print('removing last checkpoint')
                                if (os.path.exists(scratch_folder + '/chkpt')):
                                    shutil.rmtree(scratch_folder + '/chkpt')
                                else:
                                    print('folder ' + scratch_folder + '/chkpt' + ' does not exist, skipping')

                        break

                    if args.make_plots:
                        plt.clf()

    # NOTTRAIN: if not doing training
    else:

        # number of variables that can be trained in the saved DNN
        tf_trainable_vars = 0

        # open the results for each trial
        for trial in range(args.nb_trials):

            # the result and scratch folders (here the same since we save to 
            # scratch only, change if needed)
            result_folder = scratchdir + '/results/' + unique_run_ID + '/' + key + '/trial_' + str(trial)
            scratch_folder = scratchdir + '/results/' + unique_run_ID + '/' + key + '/trial_' + str(trial)

            # reset the graph to initial state
            tf.reset_default_graph()

            # open the session for testing
            with tf.Session() as sess:

                if not args.quiet:
                    print("Loading run \"%s\" trial: %d from %s" % (unique_run_ID, trial, result_folder))

                # TODO: fix this method as has been deprecated
                tf.compat.v1.saved_model.load(sess, [tag_constants.SERVING], result_folder + '/final') 

                # load the graph from this saved file
                graph = tf.get_default_graph()

                # print out the number of parameters to be trained
                tf_trainable_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

                if not args.quiet:
                    print('This model has ' + str(tf_trainable_vars) + ' trainable parameters')

                # for recording the max weight
                max_weight = 0.0

                # find the max weight over this trial
                for v in tf.trainable_variables():

                    max_v = tf.reduce_max(tf.abs(v))
                    max_v_val = sess.run(max_v)

                    # update if larger 
                    if max_v_val > max_weight:
                        max_weight = max_v_val

                # record the maximum for this trial
                DNN_max_weights_trials[trial] = max_weight
                print('max weight of this trial = ' + str(max_weight))

                # get the input and output tensors of the trained DNN model by their names
                x = graph.get_tensor_by_name('Graph/input:0')
                y = graph.get_tensor_by_name('Graph/UniversalApproximator/output:0')

                # define y_true for testing
                if not args.MATLAB_data and args.input_dim == 1:
                    y_true = func_to_approx(x, args.example)
                else:
                    y_true = tf.placeholder(precision, shape = [args.output_dim, None], name = 'y_true')


                # evaluate the model's predictions on the testing set
                y_true_test_pred, y_DNN_test_pred = sess.run([y_true, y], feed_dict = {x: x_in_test, y_true: y_true_test})

                # evaluate the model's predictions on the training set 
                y_true_train_pred, y_DNN_train_pred = sess.run([y_true, y], feed_dict = {x: x_in_train, y_true: y_true_train})

                # compute statistics
                absdiff = abs(y_true_test - y_DNN_test_pred);

                # compute percentages as above
                perc_e1 = (1.0*(absdiff > 10)).sum()/args.nb_test_points*100.0;
                perc_e0 = (1.0*(absdiff > 1)).sum()/args.nb_test_points*100.0;
                perc_em1 = (1.0*(absdiff > 1e-1)).sum()/args.nb_test_points*100.0;
                perc_em2 = (1.0*(absdiff > 1e-2)).sum()/args.nb_test_points*100.0;
                perc_em3 = (1.0*(absdiff > 1e-3)).sum()/args.nb_test_points*100.0;
                perc_em4 = (1.0*(absdiff > 1e-4)).sum()/args.nb_test_points*100.0;
                perc_em5 = (1.0*(absdiff > 1e-5)).sum()/args.nb_test_points*100.0;
                perc_em6 = (1.0*(absdiff > 1e-6)).sum()/args.nb_test_points*100.0;

                # the l-infinity error on the testing points
                linferr = np.amax(absdiff);

                # the l-2 error on the testing points
                l2err = np.sqrt(sum(np.square(absdiff)))

                # the L2 error on the testing points computed with a SG quadrature rule
                L2err = np.sqrt(abs(np.sum(np.square(absdiff)*quadrature_weights_test*2.0**(-1.0*args.input_dim))))

                if not args.quiet:
                    print("saved model \"%s\" trial %d has error: linf = %6.4f, L2 = %6.4f, %%>1ek: " \
                        "0 = %6.4f, -1 = %6.4f, -2 = %6.4f, -3 = %6.4f, -4 = %6.4f, -5 = %6.4f, -6 = %6.4f"
                        % (unique_run_ID, trial, linferr, L2err, perc_e0, perc_em1, perc_em2, perc_em3, 
                            perc_em4, perc_em5, perc_em6));

                # assign to running stats
                if (trial == 0):
                    percs = [perc_e1, perc_e0, 
                            perc_em1, perc_em2, 
                            perc_em3, perc_em4, 
                            perc_em5, perc_em6]
                    y_true_test_pred = y_true_test_pred
                    y_true_train_pred_trials = y_true_train 
                    y_DNN_test_pred_trials = y_DNN_test_pred 
                    y_DNN_train_pred_trials = y_DNN_train_pred 
                else:
                    percs = np.vstack([percs, [perc_e1, perc_e0, 
                                            perc_em1, perc_em2, 
                                            perc_em3, perc_em4, 
                                            perc_em5, perc_em6]])
                    y_true_train_pred_trials = np.vstack([y_true_train_pred_trials, y_true_train])
                    y_DNN_test_pred_trials = np.vstack([y_DNN_test_pred_trials, y_DNN_test_pred])
                    y_DNN_train_pred_trials = np.vstack([y_DNN_train_pred_trials, y_DNN_train_pred])
                    

                if args.make_plots and args.input_dim == 1:
                    fig = plt.figure(figsize = (30, 10))

                    # plot error of approximation
                    plt.subplot(131)
                    plt.title('true vs. approximation')
                    plt.plot(x_in_test, y_true_test_pred, 'r')
                    plt.plot(x_in_test, y_DNN_test_pred, 'g--')

                    #plt.ylim(-1,6)

                    # plot approximation
                    plt.subplot(132)
                    plt.title('approximation')
                    plt.plot(x_in_test, y_DNN_test_pred)

                    # plot absolute error on log scale
                    plt.subplot(133)
                    plt.title('abs error')
                    plt.semilogy(x_in_test, abs(y_true_test_pred - y_DNN_test_pred))
                    #bottom = min(abs(y_true_test_pred - y_DNN_test_pred))
                    plt.ylim(1E-8,1E1)

                    plt.tight_layout()
                    plt.draw() # update after every subplot
                    #plt.show() # show the plot

                    if not os.path.exists(scratchdir + '/results/' + unique_run_ID + '/' + key + '/trained_imgs/1e6_points/'):
                        os.makedirs(scratchdir + '/results/' + unique_run_ID + '/' + key + '/trained_imgs/1e6_points/')
                    plt.savefig(scratchdir + '/results/' + unique_run_ID + '/' + key + '/trained_imgs/5e5_points/'  \
                                        + key + '_trial_' + str(trial) + '.png')
                    plt.close(fig)
                    #plt.clf()

                sys.stdout.flush()

    # AFTERTEST: after testing all the trials, compute ensemble statistics
    if not args.train:

        # the final L2 error sum
        L2_err_sum = 0.0
        # the L2 errors of each trial
        L2_errs_trials = np.zeros(args.nb_trials)

        # iterate over the trials
        for j in range(args.nb_trials):

            # flush the output
            sys.stdout.flush()

            # get the result of the j-th trial
            y_DNN_test_pred_trial_j = y_DNN_test_pred_trials[j]

            # the error of the j-th trial
            L2_err_trial_j = 0.0;

            if not args.MATLAB_data and args.input_dim == 1:
                """ #uniform approx
                L2_err_sum = 0.0

                for i in range(args.nb_test_points):
                    L2_err_sum = L2_err_sum + ((y_true_res_testing[i] - avg_y_res[i])**2)*2/args.nb_test_points

                L2_err_sum = L2_err_sum**(0.5)
                """

                #""" #trapezoidal approx        
                L2_err_trial_j = abs(y_true_test[0] - y_DNN_test_pred_trial_j[0])**2

                for i in range(args.nb_test_points-1):
                    L2_err_trial_j = L2_err_trial_j + 2.0*abs(y_true_test[i+1] - y_DNN_test_pred_trial_j[i+1])**2

                L2_err_trial_j = L2_err_trial_j + abs(y_true_test[args.nb_test_points-1] - y_DNN_test_pred_trial_j[args.nb_test_points-1])**2
                L2_err_trial_j = L2_err_trial_j*2.0/args.nb_test_points/2.0
                L2_err_trial_j = L2_err_trial_j**(0.5)
                L2_errs_trials[j] = L2_err_trial_j
                #"""
                
                """ #simpson's rule
                L2_err_trial_j = abs(y_true_test_pred[0] - y_DNN_test_pred_trial_j[0])**2

                for i in range(args.nb_test_points-1):
                    if i % 2 == 1:
                        L2_err_trial_j = L2_err_trial_j + 4.0*abs(y_true_test_pred[i+1] - y_DNN_test_pred_trial_j[i+1])**2
                    elif i % 2 == 0:
                        L2_err_trial_j = L2_err_trial_j + 2.0*abs(y_true_test_pred[i+1] - y_DNN_test_pred_trial_j[i+1])**2

                L2_err_trial_j = L2_err_trial_j + abs(y_true_test_pred[args.nb_test_points-1] - y_DNN_test_pred_trial_j[args.nb_test_points-1])**2
                L2_err_trial_j = L2_err_trial_j*2.0/args.nb_test_points/3.0
                L2_errs_trials[j] = L2_err_trial_j
                """

                errfmt = "%6.8e" % L2_err_trial_j
                print('L2 error of trial ' + str(j).zfill(3) + ': ' + errfmt)

            # in higher dimension, we need to use a sparse grid rule to approx.
            # the L2 error, which requires the testing data from MATLAB
            else:

                # reshape the data to match y true's data
                y_DNN_test_pred_trial_j = np.array(y_DNN_test_pred_trial_j).reshape(args.output_dim,args.nb_test_points)

                # compute the absolute difference on the testing points
                absdiff_trial_j = abs(y_DNN_test_pred_trial_j - y_true_test_pred)

                # the L2 error is computed with the sparse grid quadrature rule
                L2_err_trial_j = np.sqrt(abs(np.sum(np.square(absdiff_trial_j)*quadrature_weights_test*2.0**(-1.0*args.input_dim))))
                # store the result 
                L2_errs_trials[j] = L2_err_trial_j

                errfmt = "%6.8e" % L2_err_trial_j
                print('L2 error of trial ' + str(j).zfill(3) + ': ' + errfmt)

            # add the error to the running sum of errors
            L2_err_sum = L2_err_sum + L2_err_trial_j

        # compute the average over the trials
        avg_L2_err = L2_err_sum/args.nb_trials

        # compute the average absolute maximum of the weights and biases over the trials
        DNN_max_weight_avg = np.sum(DNN_max_weights_trials)/args.nb_trials

        print("Final average L2 error: %8.16e" % (avg_L2_err))
        print("Final average max weight: %8.4f" % (DNN_max_weight_avg))

        # save the ensemble data for plotting
        ensemble_data = {}
        ensemble_data['avg_L2_err'] = avg_L2_err
        ensemble_data['L2_errs_trials'] = L2_errs_trials
        ensemble_data['DNN_max_weights_trials'] = DNN_max_weights_trials
        ensemble_data['DNN_max_weight_avg'] = DNN_max_weight_avg
        ensemble_data['y_DNN_test_pred_trials'] = y_DNN_test_pred_trials
        ensemble_data['y_DNN_train_pred_trials'] = y_DNN_train_pred_trials
        ensemble_data['y_true_test_pred'] = y_true_test_pred
        ensemble_data['y_true_train_trials'] = y_true_train_trials
        ensemble_data['x_in_test'] = x_in_test
        ensemble_data['x_in_train_trials'] = x_in_train_trials
        ensemble_data['tf_trainable_vars'] = tf_trainable_vars
        ensemble_data['sigma'] = sigma

        # save the ensemble data as a MATLAB -v7.3 hdf5-compatible mat file
        sio.savemat(scratchdir + '/results/' + unique_run_ID + '/' + key + '/ensemble_data.mat', ensemble_data)

        if args.make_plots and args.input_dim == 1:
            plt.clf()
            plt.plot(x_in_test, y_true_res_testing, 'r')
            plt.plot(x_in_test, avg_y_res, 'g--')
            plt.draw()
            plt.show()
