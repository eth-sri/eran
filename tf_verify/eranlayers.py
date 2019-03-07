'''
@author: Adrian Hoffmann
'''
import tensorflow as tf
import numpy as np
from functools import reduce


def tensorshape_to_intlist(tensorshape):
    """
    TensorFlow has its own wrapper for shapes because some could be None. This function turns them into int-lists. None will become a 1.
    
    Arguments
    ---------
    tensorshape : tf.TensorShape
    
    Return
    ------
    output : list
        list of ints corresponding to tensorshape
    """
    return list(map(lambda j: 1 if j.value is None else int(j), tensorshape))



def eran_affine(inputs, units, name=None):
    """
    adds a dense layer without activation to the graph (including bias). If inputs doesn't have a fitting shape it will be reshaped.
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer  
    units : int
        number of neurons this layer maps to
    name : str 
        optional name for the bias add operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the bias add operation
    """
    n      = reduce(lambda x,y: x*y, tensorshape_to_intlist(inputs.shape), 1)
    inputs = tf.reshape(inputs, (-1, n))
    matrix = tf.Variable(tf.glorot_uniform_initializer(dtype=tf.float64)((n, units)))
    bias   = tf.Variable(tf.zeros_initializer(dtype=tf.float64)((units,)))
    
    model = tf.matmul(inputs, matrix)
    return tf.nn.bias_add(model, bias, name=name)


def eran_conv2d_without_activation(inputs, kernel_size, number_of_filters, strides, padding, name=None):
    """
    adds a convolutional layer without activation to the graph (including bias).
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer (needs to be a 4D tensor)
    kernel_size : list or tuple
        two ints, specifying height and width of the kernel
    number_of_filters : int
        number of filters this convolutional layer should have
    strides : list or tuple
        two ints, specifying the stride in height and width direction
    padding : str
        either "VALID" or "SAME", see TensorFlow documentation for further information
    name : str
        optional name for the bias add operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the bias add operation
    """
    assert padding in ['SAME', 'VALID'], "padding must be either 'VALID' or 'SAME'"
    
    in_channels = int(inputs.shape[-1])
    filters     = tf.Variable(tf.glorot_uniform_initializer(dtype=tf.float64)((kernel_size[0], kernel_size[1], in_channels, number_of_filters)))
    bias        = tf.Variable(tf.zeros_initializer(dtype=tf.float64)((number_of_filters,)))
    
    model = tf.nn.conv2d(inputs, filters, [1, strides[0], strides[1], 1], padding)
    return tf.nn.bias_add(model, bias, name=name)


def eran_activation(inputs, activation, name=None):
    """
    applies a non-linearity to inputs
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer
    activation : str
        one of ['relu', 'sigmoid', 'tanh']
    name : str
        optional name for the non-linearity operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the non-linearity operation 
    """
    assert activation in ['relu', 'sigmoid', 'tanh'], "the activation has to be in ['relu', 'sigmoid', 'tanh']"
    if activation == 'relu':
        activation = tf.nn.relu
    elif activation == 'sigmoid':
        activation = tf.sigmoid
    elif activation == 'tanh':
        activation = tf.tanh
    return activation(inputs, name=name)


def eran_input(shape, name=None):
    """
    adds a tf.Placeholder to the graph. The shape will be augmented with None at the beginning as batch size
    
    Arguments
    ---------
    shape : list or tuple
        the shape of the Placeholder, has 1 to 3 entries
    name : str
        optional name for the Placeholder operation  
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the Placeholder operation
    """
    assert len(shape) < 4, "shape should have less than 4 entries (batch size is taken care of)"
    batch_shape = [None]
    for s in shape:
        batch_shape.append(s)
    
    return tf.placeholder(tf.float64, batch_shape, name=name)


def eran_reshape(inputs, new_shape, name=None):
    """
    adds a tf.Reshape to the graph. Inputs will be reshaped to (-1,)+new_shape
    
    Arguments
    ---------
    inputs : tf.Tensor
        the preceding layer
    new_shape : list or tuple
        the new shape, has 1 to 3 entries
    name : str
        optional name for the Reshape operation
    
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the Reshape operation 
    """
    assert len(new_shape) < 4, "shape should have less than 4 entries (batch size is taken care of)"
    batch_shape = [-1]
    for s in new_shape:
        batch_shape.append(s)
    return tf.reshape(inputs, batch_shape, name=name)    


def eran_dense(inputs, units, activation='relu', name=None):
    """
    adds a dense layer to the graph, including bias
    
    Arguments
    ---------
    inputs : tf.Tensor
        the preceding layer
    units : int
        the number of neurons this dense layer maps to
    activation : str
        one of ['relu', 'sigmoid', 'tanh'], default is 'relu'
    name : str
        optional name for the non-linearity operation at the end
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the non-linearity operation
    """
    tmp  = eran_affine(inputs, units)
    return eran_activation(tmp, activation, name=name)


def eran_conv2d(inputs, kernel_size, number_of_filters, strides, padding, activation='relu', name=None):
    """
    adds a convolution layer to the graph, including bias
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer (needs to be a 4D tensor)
    kernel_size : list or tuple
        two ints, specifying height and width of the kernel
    number_of_filters : int
        number of filters this convolutional layer should have
    strides : list or tuple
        two ints, specifying the stride in height and width direction
    padding : str
        either "VALID" or "SAME", see TensorFlow documentation for further information
    activation : str
        one of ['relu', 'sigmoid', 'tanh'], default is 'relu'
    name : str
        optional name for the bias add operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the non-linearity operation
    """
    tmp  = eran_conv2d_without_activation(inputs, kernel_size, number_of_filters, strides, padding)
    return eran_activation(tmp, activation, name=name)


def eran_maxpool(inputs, pool_size, strides, padding, name=None):
    """
    adds a MaxPool layer to the graph
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer (needs to be a 4D tensor)
    pool_size : list or tuple
        two ints, specifying height and width of the window
    number_of_filters : int
        number of filters this convolutional layer should have
    strides : list or tuple
        two ints, specifying the stride in height and width direction
    padding : str
        either "VALID" or "SAME", see TensorFlow documentation for further information
    name : str
        optional name for the MaxPool operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the MaxPool operation
    """
    assert padding in ['SAME', 'VALID'], "padding must be either 'VALID' or 'SAME'"
    assert len(inputs.shape) == 4, "inputed tensor isn't an image (4D tensor)"
    
    return tf.nn.max_pool(inputs, [1, pool_size[0], pool_size[1], 1], [1, strides[0], strides[1], 1], padding, name=name)


def eran_resnet_conv2d(inputs, kernel_size, number_of_layers, activation='relu', name=None):
    """
    Adds a specified number of convolution layers to the graph. Each of the layers has a stride of one in each direction. Padding is always "SAME".
    Additionally there will be an add connection. The information in inputs will be added to the information of the last convolution layer right
    after the bias-add but before the application of the non-linearity.
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer (needs to be a 4D tensor)
    kernel_size : list or tuple
        two ints, specifying height and width of the kernel. Is the same for each layer.
    number_of_layers : int
        number of convolution layers that should be added
    activation : str
        one of ['relu', 'sigmoid', 'tanh']. Is the same for each layer. Default is 'relu'.
    name : str
        optional name for the last non-linearity operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the last non-linearity operation
    """
    assert number_of_layers > 0, "number_of_layers must be greater than zero"
    number_of_filters = int(inputs.shape[-1])
    tmp = inputs
    for i in range(number_of_layers-1):
        tmp = eran_conv2d(tmp, kernel_size, number_of_filters, [1,1], 'SAME', activation)
    tmp = eran_conv2d_without_activation(tmp, kernel_size, number_of_filters, [1,1], 'SAME')
    tmp = tf.add(tmp, inputs)
    return eran_activation(tmp, activation, name=name)


def eran_resnet_dense(inputs, number_of_layers, activation='relu', name=None):
    """
    Adds a specified number of dense layers to the graph. Each layer maps to the same number of neurons as inputs has. If inputs doesn't have a fitting shape it will be reshaped.
    Additionally there will be an add connection. The information in inputs will be added to the information of the last dense layer right after the bias-add but before the application 
    of the non-linearity.
    
    Arguments
    ---------
    inputs : tf.Tensor
        preceding layer
    number_of_layers : int
        number of dense layers that should be added
    activation : str
        one of ['relu', 'sigmoid', 'tanh']. Is the same for each layer. Default is 'relu'.
    name : str
        optional name for the last non-linearity operation
    
    Return
    ------
    output : tf.Tensor
        tensor associated with the last non-linearity operation
    """
    assert number_of_layers > 0, "number_of_layers must be greater than zero"
    units  = reduce(lambda x,y: x*y, tensorshape_to_intlist(inputs.shape), 1)
    inputs = eran_reshape(inputs, [units])
    tmp    = inputs
    for i in range(number_of_layers-1):
        tmp = eran_dense(tmp, units, activation)
    tmp = eran_affine(tmp, units)
    tmp = tf.add(tmp, inputs)
    return eran_activation(tmp, activation, name=name)

