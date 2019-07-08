'''
@author: Adrian Hoffmann
'''



import numpy as np
from fppoly import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from functools import reduce

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
    self.output_length = reduce((lambda x, y: x*y), output_shape)
    self.input_names   = input_names
    self.output_name   = output_name


class DeeppolyInput:
    def __init__(self, specLB, specUB, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec
        """
        self.specLB = np.ascontiguousarray(specLB, dtype=np.double)
        self.specUB = np.ascontiguousarray(specUB, dtype=np.double)
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
        return fppoly_from_network_input(man, 0, len(self.specLB), self.specLB, self.specUB)



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
        return xpp, self.bias, self.weights.shape[0], self.weights.shape[1], self.predecessors
    
    
    def get_xpp(self):
        """
        helper function to get pointers to the rows of self.weights.
        
        Return
        ------
        output : numpy.ndarray
            pointers to the rows of the matrix
        """
        return (self.weights.__array_interface__['data'][0]+ np.arange(self.weights.shape[0])*self.weights.strides[0]).astype(np.uintp)




class DeeppolyReluNodeFirst(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
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
        ffn_handle_first_relu_layer(man, element, *self.get_arguments())
        bounds = box_for_layer(man, element, nn.ffn_counter+nn.conv_counter)
        num_neurons = get_num_neurons_in_layer(man, element, nn.ffn_counter+nn.conv_counter)
        lbi = []
        ubi = []
        for i in range(num_neurons):
            inf = bounds[i].contents.inf
            sup = bounds[i].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 
        
        elina_interval_array_free(bounds,num_neurons)
        nn.ffn_counter+=1
        return element


class DeeppolySigmoidNodeFirst(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
            transformer for the first layer of a neural network, if that first layer is fully connected with sigmoid
            
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
        ffn_handle_first_sigmoid_layer(man, element, *self.get_arguments())
        return element


class DeeppolyTanhNodeFirst(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
            transformer for the first layer of a neural network, if that first layer is fully connected with tanh
            
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
        ffn_handle_first_tanh_layer(man, element, *self.get_arguments())
        return element



class DeeppolyReluNodeIntermediate(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
        transformer for any intermediate fully connected layer with relu
        
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
        ffn_handle_intermediate_relu_layer(man, element, *self.get_arguments(), use_area_heuristic)
        bounds = box_for_layer(man, element, nn.ffn_counter+nn.conv_counter)
        num_neurons = get_num_neurons_in_layer(man, element, nn.ffn_counter+nn.conv_counter)
        lbi = []
        ubi = []
        for i in range(num_neurons):
            inf = bounds[i].contents.inf
            sup = bounds[i].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 
        
        elina_interval_array_free(bounds,num_neurons)
        nn.ffn_counter+=1
        return element

class DeeppolySigmoidNodeIntermediate(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
            transformer for any intermediate fully connected layer with sigmoid
            
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
        ffn_handle_intermediate_sigmoid_layer(man, element, *self.get_arguments(), use_area_heuristic)
        return element


class DeeppolyTanhNodeIntermediate(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
            transformer for any intermediate fully connected layer with tanh
            
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
        ffn_handle_intermediate_tanh_layer(man, element, *self.get_arguments(), use_area_heuristic)
        return element



class DeeppolyReluNodeLast(DeeppolyNode):
    def __init__(self, weights, bias, relu_present, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        weights : numpy.ndarray
            matrix of the fully connected layer (must be 2D)
        bias : numpy.ndarray
            bias of the fully connected layer
        relu_present : bool
            whether this layer has relu or not
        """
        DeeppolyNode.__init__(self, weights, bias, input_names, output_name, output_shape)
        self.relu_present = relu_present
        
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
        transformer for a fully connected layer if it's the last layer in the network
        
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
        ffn_handle_last_relu_layer(man, element, *self.get_arguments(), self.relu_present, use_area_heuristic)
        bounds = box_for_layer(man, element, nn.ffn_counter+nn.conv_counter)
        num_neurons = get_num_neurons_in_layer(man, element, nn.ffn_counter+nn.conv_counter)
        lbi = []
        ubi = []
        for i in range(num_neurons):
            inf = bounds[i].contents.inf
            sup = bounds[i].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 
        
        elina_interval_array_free(bounds,num_neurons)
        nn.ffn_counter+=1
        return element


class DeeppolySigmoidNodeLast(DeeppolyNode):
    def __init__(self, weights, bias, sigmoid_present, input_names, output_name, output_shape):
        """
            Arguments
            ---------
            weights : numpy.ndarray
            matrix of the fully connected layer (must be 2D)
            bias : numpy.ndarray
            bias of the fully connected layer
            relu_present : bool
            whether this layer has sigmoid or not
            """
        DeeppolySigmoidNode.__init__(self, weights, bias, input_names, output_name, output_shape)
        self.sigmoid_present = sigmoid_present
            
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
                    transformer for a fully connected layer if it's the last layer in the network
                    
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
        ffn_handle_last_sigmoid_layer(man, element, *self.get_arguments(), self.sigmoid_present, use_area_heuristic)
        return element


class DeeppolyTanhNodeLast(DeeppolyNode):
    def __init__(self, weights, bias, tanh_present, input_names, output_name, output_shape):
        """
            Arguments
            ---------
            weights : numpy.ndarray
            matrix of the fully connected layer (must be 2D)
            bias : numpy.ndarray
            bias of the fully connected layer
            relu_present : bool
            whether this layer has relu or not
            """
        DeeppolyTanhNode.__init__(self, weights, bias, input_names, output_name, output_shape)
        self.tanh_present = tanh_present
            
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
                    transformer for a fully connected layer if it's the last layer in the network
                    
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
        ffn_handle_last_tanh_layer(man, element, *self.get_arguments(), self.tanh_present, use_area_heuristic)
        return element


class DeeppolyConv2dNodeIntermediate:
    def __init__(self, filters, strides, padding, bias, image_shape, input_names, output_name, output_shape):
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
        self.padding    =  padding
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
        return self.filters, self.bias, self.image_shape, filter_size, numfilters, strides, self.padding == "VALID", True, self.predecessors
        
            
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
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
        print("predecessors ", self, self.predecessors) 
        conv_handle_intermediate_relu_layer(man, element, *self.get_arguments(), use_area_heuristic)
        bounds = box_for_layer(man, element, nn.ffn_counter+nn.conv_counter)
        num_neurons = get_num_neurons_in_layer(man, element, nn.ffn_counter+nn.conv_counter)
        lbi = []
        ubi = []
        for i in range(num_neurons):
            inf = bounds[i].contents.inf
            sup = bounds[i].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 
        
        elina_interval_array_free(bounds,num_neurons)
        nn.conv_counter+=1
        return element




class DeeppolyConv2dNodeFirst(DeeppolyConv2dNodeIntermediate):    
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
        transformer for a convolutional layer, if that layer is the first of the network
        
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
        conv_handle_first_layer(man, element, *self.get_arguments())
        bounds = box_for_layer(man, element, nn.ffn_counter+nn.conv_counter)
        num_neurons = get_num_neurons_in_layer(man, element, nn.ffn_counter+nn.conv_counter)
        lbi = []
        ubi = []
        for i in range(num_neurons):
            inf = bounds[i].contents.inf
            sup = bounds[i].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 
        
        elina_interval_array_free(bounds,num_neurons)
        nn.ffn_counter+=1
        return element    




class DeeppolyMaxpool:
    def __init__(self, image_shape, window_size, strides, input_names, output_name, output_shape):
        """
        collects the information needed for the handle_maxpool_layer transformer and brings it into the required shape
        
        Arguments
        ---------
        input_shape : numpy.ndarray
            1D array of ints with 3 entries [height, width, channels] representing the shape of the of the image that is passed to the conv-layer
        window_size : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of ints with 2 entries [height, width] representing the stride in these directions
        """
        self.image_shape = np.ascontiguousarray(image_shape, dtype=np.uintp)
        self.window_size = np.ascontiguousarray([window_size[0], window_size[1], 1], dtype=np.uintp)
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    
    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        """
        transformer for a maxpool layer, this can't be the first layer of a network
        
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
        handle_maxpool_layer(man, element, self.window_size, self.image_shape, self.predecessors)
        return element


class DeeppolyResadd:
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

    def transformer(self, nn, man, element, nlb, nub, use_area_heuristic):
        handle_residual_layer(man,element,self.output_length,self.predecessors)
        return element
