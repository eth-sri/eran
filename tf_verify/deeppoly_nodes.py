"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import numpy as np
from config import config, Device

if config.device == Device.CPU:
    from fppoly import *
else:
    from fppoly_gpu import *

from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from ai_milp import *
from functools import reduce
from refine_activation import *


def calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer = False, destroy=True, use_krelu = False):
    layerno = nn.calc_layerno()
    bounds = box_for_layer(man, element, layerno)
    num_neurons = get_num_neurons_in_layer(man, element, layerno)
    itv = [bounds[i] for i in range(num_neurons)]
    lbi = [x.contents.inf.contents.val.dbl for x in itv]
    ubi = [x.contents.sup.contents.val.dbl for x in itv]
    if is_refine_layer:
        nlb.append(lbi)
        nub.append(ubi)
    if destroy:
        elina_interval_array_free(bounds,num_neurons)
        return lbi, ubi
    return layerno, bounds, num_neurons, lbi, ubi


def add_input_output_information_deeppoly(self, input_names, output_name, output_shape):
    """
    sets for an object the three fields:
        - self.output_length
        - self.input_names
        - self.output_name
    which will mainly be used by the Optimizer, but can also be used by the Nodes itself
    
    Arguments
    ---------
    self : Object
        will be a DeepzonoNode, but could be any object
    input_names : iterable
        iterable of strings, each one being the name of another Deepzono-Node
    output_name : str
        name of self
    output_shape : iterable
        iterable of ints with the shape of the output of this node
        
    Return
    ------
    None 
    """
    if len(output_shape)==4:
        self.output_length = reduce((lambda x, y: x*y), output_shape[1:len(output_shape)])
    else:
        self.output_length = reduce((lambda x, y: x*y), output_shape[0:len(output_shape)])
    self.input_names   = input_names
    self.output_name   = output_name


class DeeppolyInput:
    def __init__(self, specLB, specUB, input_names, output_name, output_shape,
                 lexpr_weights=None, lexpr_cst=None, lexpr_dim=None,
                 uexpr_weights=None, uexpr_cst=None, uexpr_dim=None,
                 expr_size=0, spatial_constraints=None):
        """
        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec
        lexpr_weights: numpy.ndarray
            ndarray of doubles with coefficients of lower polyhedral expressions
        lexpr_cst: numpy.ndarray
            ndarray of doubles with the constants of lower polyhedral expressions
        lexpr_dim: numpy.ndarray
            ndarray of unsigned int with the indexes of pixels from the original image for the lower polyhedral expressions
        uexpr_weights: numpy.ndarray
            ndarray of doubles with coefficients of upper polyhedral expressions
        uexpr_cst: numpy.ndarray
            ndarray of doubles with the constants of upper polyhedral expressions
        uexpr_dim: numpy.ndarray
            ndarray of unsigned int with the indexes of pixels from the original image for the upper polyhedral expressions
        expr_size: numpy.ndarray
            unsigned int with the sizes of polyhedral expressions
        """
        self.specLB = np.ascontiguousarray(specLB, dtype=np.double)
        self.specUB = np.ascontiguousarray(specUB, dtype=np.double)

        if lexpr_weights is not None:
            self.lexpr_weights = np.ascontiguousarray(lexpr_weights, dtype=np.double)
        else:
            self.lexpr_weights = None
        if lexpr_cst is not None:
            self.lexpr_cst = np.ascontiguousarray(lexpr_cst, dtype=np.double)
        else:
            self.lexpr_cst = None
        if lexpr_dim is not None:
            self.lexpr_dim = np.ascontiguousarray(lexpr_dim, dtype=np.uintp)
        else:
            self.lexpr_dim = None

        if uexpr_weights is not None:
            self.uexpr_weights = np.ascontiguousarray(uexpr_weights, dtype=np.double)
        else:
            self.uexpr_weights = None
        if uexpr_cst is not None:
            self.uexpr_cst = np.ascontiguousarray(uexpr_cst, dtype=np.double)
        else:
            self.uexpr_cst = None
        if uexpr_dim is not None:
            self.uexpr_dim = np.ascontiguousarray(lexpr_dim, dtype=np.uintp)
        else:
            self.uexpr_dim = None

        self.expr_size = expr_size

        self.spatial_gamma = -1
        self.spatial_indices = np.ascontiguousarray([], np.uint64)
        self.spatial_neighbors = np.ascontiguousarray([], np.uint64)

        if spatial_constraints is not None:
            self.spatial_gamma = spatial_constraints['gamma']
            self.spatial_indices = np.ascontiguousarray(
                spatial_constraints['indices'], np.uint64
            )
            self.spatial_neighbors = np.ascontiguousarray(
                spatial_constraints['neighbors'], np.uint64
            )

        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)


    def transformer(self, man):
        """
        creates an abstract element from the input spec
        
        Arguments
        ---------
        man : ElinaManagerPtr
            inside this manager the abstract element will be created
        
        Return
        ------
        output : ElinaAbstract0Ptr
            new abstract element representing the element specified by self.specLB and self.specUB
        """
        if self.expr_size == 0:
            return fppoly_from_network_input(man, 0, len(self.specLB), self.specLB, self.specUB)
        else:
            return fppoly_from_network_input_poly(
                man, 0, len(self.specLB), self.specLB, self.specUB,
                self.lexpr_weights, self.lexpr_cst, self.lexpr_dim,
                self.uexpr_weights, self.uexpr_cst, self.uexpr_dim,
                self.expr_size, self.spatial_indices, self.spatial_neighbors,
                len(self.spatial_indices), self.spatial_gamma
            )


class DeeppolyNode:
    """
    Parent class for all the classes that implement fully connected layers
    """
    def __init__(self, weights, bias, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        weights : numpy.ndarray
            matrix of the fully connected layer (must be 2D)
        bias : numpy.ndarray
            bias of the fully connected layer
        """
        self.weights = np.ascontiguousarray(weights, dtype=np.double)
        self.bias    = np.ascontiguousarray(bias,    dtype=np.double)
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def get_arguments(self):
        """
        facilitates putting together all the arguments for the transformers in the child classes
        
        Return
        ------
        output : tuple
            the four entries are pointers to the rows of the matrix, the bias, the length of the output, and the length of the input
        """
        xpp = self.get_xpp()
        return xpp, self.bias, self.weights.shape[0], self.weights.shape[1], self.predecessors, len(self.predecessors)


    def get_xpp(self):
        """
        helper function to get pointers to the rows of self.weights.
        
        Return
        ------
        output : numpy.ndarray
            pointers to the rows of the matrix
        """
        return (self.weights.__array_interface__['data'][0]+ np.arange(self.weights.shape[0])*self.weights.strides[0]).astype(np.uintp)




class DeeppolyFCNode(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        """
        transformer for the first layer of a neural network, if that first layer is fully connected with relu
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer 
        """
        handle_fully_connected_layer(man, element, *self.get_arguments())
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, use_krelu=refine)
        nn.ffn_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolyNonlinearity:
    def __init__(self, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        input_names : iterable
            iterable with the name of the vector you want to apply the non-linearity to
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)
    
    
    def get_arguments(self, man, element):
        """
        used by the children of this class to easily get the inputs for their transformers
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : tuple
            arguments for the non-linearity transformers like Relu or Sigmoid 
        """
        length = self.output_length
        return man, element, length, self.predecessors, len(self.predecessors)




class DeeppolyReluNode(DeeppolyNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing, K=3, s=-2, use_milp=False, approx=True):
        """
        transforms element with handle_relu_layer
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        length = self.output_length
        if refine:
            refine_activation_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp,
                                                 timeout_milp, use_default_heuristic, 'deeppoly',
                                                 K=K, s=s, use_milp=use_milp, approx=approx)
        else:
            handle_relu_layer(*self.get_arguments(man, element), use_default_heuristic)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, use_krelu=False)
        nn.activation_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]

        return element
 

class DeeppolySignNode(DeeppolyNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp,
                    use_default_heuristic, testing, K=3, s=-2, approx=True):
        """
        transforms element with handle_sign_layer
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        #if refine:
        #    refine_activation_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp, timeout_milp, use_default_heuristic, 'deeppoly')
        #else:
        handle_sign_layer(*self.get_arguments(man, element))
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, use_krelu=False)
        nn.activation_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]

        return element


class DeeppolySigmoidNode(DeeppolyNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing, K=3, s=-2, use_milp=False, approx=True):
        """
        transforms element with handle_sigmoid_layer
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        length = self.output_length
        if refine:
            refine_activation_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp, timeout_milp, use_default_heuristic, 'deeppoly', K=K, s=s, use_milp=use_milp)
        else:
            handle_sigmoid_layer(*self.get_arguments(man, element), use_default_heuristic)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, use_krelu=refine)
        nn.activation_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]

        return element
        
        
class DeeppolyTanhNode(DeeppolyNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing, K=3, s=-2, use_milp=False, approx=True):
        """
        transforms element with handle_tanh_layer
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        length = self.output_length
        
        if refine:
            refine_activation_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp, timeout_milp, use_default_heuristic, 'deeppoly',K=K, s=s, use_milp=use_milp)
        else:
            handle_tanh_layer(*self.get_arguments(man, element), use_default_heuristic)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, use_krelu=refine)
        nn.activation_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]

        return element


class DeeppolyLeakyReluNode(DeeppolyNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing, alpha=0.01):
        """
        transforms element with handle_tanh_layer
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        length = self.output_length
        
        if False:
            refine_activation_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp, timeout_milp, use_default_heuristic, 'deeppoly')
        else:
            handle_leakyrelu_layer(*self.get_arguments(man, element), alpha, use_default_heuristic)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, use_krelu=refine)
        nn.activation_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]

        return element

class DeeppolyConv2dNode:
    def __init__(self, filters, strides, pad_top, pad_left, pad_bottom, pad_right, bias, image_shape, input_names, output_name, output_shape):
        """
        collects the information needed for the conv_handle_intermediate_relu_layer transformer and brings it into the required shape
        
        Arguments
        ---------
        filters : numpy.ndarray
            the actual 4D filter of the convolutional layer
        strides : numpy.ndarray
            1D with to elements, stride in height and width direction
        bias : numpy.ndarray
            the bias of the layer
        image_shape : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        """
        self.image_shape = np.ascontiguousarray(image_shape, dtype=np.uintp)
        self.filters     = np.ascontiguousarray(filters, dtype=np.double)
        self.strides     = np.ascontiguousarray(strides, dtype=np.uintp)
        self.bias        = np.ascontiguousarray(bias, dtype=np.double)
        self.out_size    = (c_size_t * 3)(output_shape[1], output_shape[2], output_shape[3]) 
        self.pad_top     = pad_top
        self.pad_left    = pad_left
        self.pad_bottom = pad_bottom
        self.pad_right = pad_right
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def get_arguments(self):
        """
        facilitates putting together all the arguments for the transformers in the child classes
        
        Return
        ------
        output : tuple
            the 5 entries are:
                1. the filter (numpy.ndarray)
                2. the bias (numpy.ndarray)
                3. the image_shape (numpy.ndarray)
                4. length of a side of the square kernel (int)
                5. number of filters (int)
        """
        filter_size = (c_size_t * 2) (self.filters.shape[0], self.filters.shape[1])
        numfilters  = self.filters.shape[3]
        strides     = (c_size_t * 2)(self.strides[0], self.strides[1])
        return self.filters, self.bias, self.image_shape, filter_size, numfilters, strides, self.out_size, self.pad_top, self.pad_left, self.pad_bottom, self.pad_right, True, self.predecessors, len(self.predecessors)


    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        """
        transformer for a convolutional layer, if that layer is an intermediate of the network
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer 
        """
        handle_convolutional_layer(man, element, *self.get_arguments())
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True)
        nn.conv_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolyPaddingNode:
    def __init__(self, pad_top, pad_left, pad_bottom, pad_right, image_shape, input_names,
                 output_name, output_shape):
        """
        collects the information needed for the conv_handle_intermediate_relu_layer transformer and brings it into the required shape

        Arguments
        ---------
        filters : numpy.ndarray
            the actual 4D filter of the convolutional layer
        strides : numpy.ndarray
            1D with to elements, stride in height and width direction
        bias : numpy.ndarray
            the bias of the layer
        image_shape : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        """
        self.image_shape =  np.ascontiguousarray(image_shape, dtype=np.uintp)
        self.out_size = (c_size_t * 3)(output_shape[1], output_shape[2], output_shape[3])
        self.pad_top = pad_top
        self.pad_left = pad_left
        self.pad_bottom = pad_bottom
        self.pad_right = pad_right
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def get_arguments(self):
        """
        facilitates putting together all the arguments for the transformers in the child classes

        Return
        ------
        output : tuple
            the 5 entries are:
                3. the image_shape (numpy.ndarray)
        """
        return self.image_shape, self.out_size, self.pad_top, self.pad_left, self.pad_bottom, self.pad_right, \
               self.predecessors, len(self.predecessors)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp,
                    use_default_heuristic, testing):
        """
        transformer for a convolutional layer, if that layer is an intermediate of the network

        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied

        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        handle_padding_layer(man, element, *self.get_arguments())
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True)
        nn.pad_counter += 1
        if testing:
            return element, nlb[-1], nub[-1]
        return element

class DeeppolyPoolNode:
    def __init__(self, input_shape, window_size, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, output_shape, is_maxpool):
        """
        collects the information needed for the handle_pool_layer transformer and brings it into the required shape
        
        Arguments
        ---------
        input_shape : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        window_size : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the stride in these directions
        """
        self.input_shape = np.ascontiguousarray(input_shape, dtype=np.uintp)
        self.window_size = np.ascontiguousarray(window_size, dtype=np.uintp)
        self.strides = np.ascontiguousarray(strides, dtype=np.uintp)
        self.pad_top = pad_top
        self.pad_left = pad_left
        self.pad_bottom = pad_bottom
        self.pad_right = pad_right
        self.output_shape = (c_size_t * 3)(output_shape[1],output_shape[2],output_shape[3])
        self.is_maxpool = is_maxpool
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)


    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        """
        transformer for a maxpool/averagepool layer, this can't be the first layer of a network
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer 
        """
        h, w = self.window_size
        H, W, C = self.input_shape
        #assert self.pad_top==self.pad_bottom==self.pad_right==self.pad_left==0, "Padded pooling not implemented"
        handle_pool_layer(man, element, (c_size_t *3)(h,w,1), (c_size_t *3)(H, W, C), (c_size_t *2)(self.strides[0], self.strides[1]), self.pad_top, self.pad_left, self.pad_bottom, self.pad_right, self.output_shape, self.predecessors, len(self.predecessors), self.is_maxpool)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, destroy=False)
        nn.pool_counter += 1
        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolyResidualNode:
    def __init__(self, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        input_names : iterable
            iterable with the names of the two nodes you want to add
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        handle_residual_layer(man,element,self.output_length,self.predecessors, len(self.predecessors))
        calc_bounds(man, element, nn, nlb, nub, relu_groups, use_krelu=refine, is_refine_layer=True)
        # print("Residual ", nn.layertypes[layerno],layerno)
        nn.residual_counter += 1

        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolyGather:
    def __init__(self, indexes, input_names, output_name, output_shape):
        """
        collects the information needed for the handle_gather_layer transformer and brings it into the required shape

        Arguments
        ---------
        indexes : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        window_size : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the stride in these directions
        """
        self.indexes = np.ascontiguousarray(indexes, dtype=np.uintp)
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        handle_gather_layer(man, element, self.indexes)
        return element


class DeeppolyConcat:
    def __init__(self, width, height, channels, input_names, output_name, output_shape):
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)
        self.width = width
        self.height = height
        self.channels = (c_size_t * len(channels))()
        for i, channel in enumerate(channels):
            self.channels[i] = channel


    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        handle_concatenation_layer(man, element, self.predecessors, len(self.predecessors), self.channels)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, destroy=False)
        nn.concat_counter += 1
        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolyTile:
    def __init__(self, repeats, input_names, output_name, output_shape):
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)
        self.repeats = repeats

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        handle_tiling_layer(man, element, self.predecessors, len(self.predecessors), self.repeats)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True, destroy=False)
        nn.tile_counter += 1
        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolySubNode:
    def __init__(self, bias, is_minuend, input_names, output_name, output_shape):
        """
        collects the information needed for the handle_gather_layer transformer and brings it into the required shape

        Arguments
        ---------
        indexes : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        window_size : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the stride in these directions
        """
        self.bias = np.ascontiguousarray(bias.reshape(-1), dtype=np.float64)
        self.is_minuend = is_minuend
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        layerno = nn.calc_layerno()
        num_neurons = get_num_neurons_in_layer(man, element, layerno)
        handle_sub_layer(man, element, self.bias, self.is_minuend, num_neurons, self.predecessors, len(self.predecessors))
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True)
        nn.ffn_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]
        return element


class DeeppolyMulNode:
    def __init__(self, bias, input_names, output_name, output_shape):
        """
        collects the information needed for the handle_gather_layer transformer and brings it into the required shape

        Arguments
        ---------
        indexes : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        window_size : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the stride in these directions
        """
        self.bias = np.ascontiguousarray(bias.reshape(-1), dtype=np.float64)
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_default_heuristic, testing):
        handle_mul_layer(man, element, self.bias, len(self.bias.reshape(-1)), self.predecessors, len(self.predecessors))
        calc_bounds(man, element, nn, nlb, nub, relu_groups, is_refine_layer=True)
        nn.ffn_counter+=1
        if testing:
            return element, nlb[-1], nub[-1]
        return element
