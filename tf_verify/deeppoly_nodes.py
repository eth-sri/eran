'''
@author: Adrian Hoffmann
'''



import numpy as np
from fppoly import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from deepzono_milp import *
from krelu import encode_krelu_cons
from functools import reduce
from config import config


def calc_bounds(man, element, nn, nlb, nub, relu_groups, destroy=True, use_krelu = True):
    layerno = nn.calc_layerno()
    bounds = box_for_layer(man, element, layerno)
    num_neurons = get_num_neurons_in_layer(man, element, layerno)
    itv = [bounds[i] for i in range(num_neurons)]
    lbi = [x.contents.inf.contents.val.dbl for x in itv]
    ubi = [x.contents.sup.contents.val.dbl for x in itv]
    nlb.append(lbi)
    nub.append(ubi)
    if use_krelu:
        encode_krelu_cons(nn, man, element, 0, layerno, num_neurons, lbi, ubi, relu_groups, False, 'refinepoly')
    if destroy:
        elina_interval_array_free(bounds,num_neurons)
        return
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
    self.output_length = reduce((lambda x, y: x*y), output_shape)
    self.input_names   = input_names
    self.output_name   = output_name


class DeeppolyInput:
    def __init__(self, specLB, specUB, input_names, output_name, output_shape,
                 lexpr_weights=None, lexpr_cst=None, lexpr_dim=None,
                 uexpr_weights=None, uexpr_cst=None, uexpr_dim=None,
                 expr_size=0):
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
            return fppoly_from_network_input_poly(man, 0, len(self.specLB), self.specLB, self.specUB,
                                                  self.lexpr_weights, self.lexpr_cst, self.lexpr_dim,
                                                  self.uexpr_weights, self.uexpr_cst, self.uexpr_dim, self.expr_size)



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
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        calc_bounds(man, element, nn, nlb, nub, relu_groups)
        nn.ffn_counter+=1
        return element


class DeeppolySigmoidNodeFirst(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.ffn_counter+=1
        return element


class DeeppolyTanhNodeFirst(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.ffn_counter+=1
        return element



class DeeppolyReluNodeIntermediate(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        layerno, bounds, num_neurons, lbi, ubi = calc_bounds(man, element, nn, nlb, nub, relu_groups, destroy = False, use_krelu=refine)
        candidate_vars = [i for i, (l, u) in enumerate(zip(lbi, ubi)) if l<0 and u>0]
        #print("lbi ", timeout_milp, "ubi ", timeout_lp)
        if refine:
            if layerno <= 1:
                use_milp = config.use_milp
            else:
                use_milp = 0

            if use_milp:
                timeout = timeout_milp
            else:
                timeout = timeout_lp

            #resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, layerno, layerno, num_neurons, nlb, nub, use_milp,  candidate_vars, timeout)

            #for j in indices:
            #    update_bounds_for_neuron(man,element,layerno,j,resl[j],resu[j])

            #nlb[-1] = resl
            #nub[-1] = resu



        elina_interval_array_free(bounds,num_neurons)
        nn.ffn_counter+=1
        return element


class DeeppolySigmoidNodeIntermediate(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.ffn_counter+=1
        return element


class DeeppolyTanhNodeIntermediate(DeeppolyNode):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.ffn_counter+=1
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

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        ffn_handle_last_relu_layer(man, element, *self.get_arguments(), self.relu_present,  use_area_heuristic)
        layerno, bounds, num_neurons, lbi, ubi = calc_bounds(man, element, nn, nlb, nub, relu_groups, destroy=False, use_krelu=refine)
        candidate_vars = [i for i, (l, u) in enumerate(zip(lbi, ubi)) if l<0 and u>0]

        if(refine):
            if layerno<=1:
                use_milp = 1
            else:
                use_milp = 0

            if use_milp:
                timeout = timeout_milp
            else:
                timeout = timeout_lp
            #resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, layerno, layerno, num_neurons, nlb, nub, use_milp,  candidate_vars, timeout)
            #for i in indices:
            #    update_bounds_for_neuron(man,element,layerno,j,resl[j],resu[j])

            #print("resl ", resl, "resu ", resu)
            #nlb[-1] = resl
            #nub[-1] = resu
            #encode_2reLu_cons(nn, man, element, 0, layerno, num_neurons, lbi, ubi, False, 'refinepoly')

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

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.ffn_counter+=1
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

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.ffn_counter+=1
        return element


class DeeppolyConv2dNodeIntermediate:
    def __init__(self, filters, strides, pad_top, pad_left, bias, image_shape, input_names, output_name, output_shape, has_relu):
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
        self.has_relu    = has_relu
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
        return self.filters, self.bias, self.image_shape, filter_size, numfilters, strides, self.out_size, self.pad_top, self.pad_left, True, self.predecessors


    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        if(self.has_relu):
            conv_handle_intermediate_relu_layer(man, element, *self.get_arguments(), use_area_heuristic)
        else:
            conv_handle_intermediate_affine_layer(man, element, *self.get_arguments(), use_area_heuristic)
        layerno, bounds, num_neurons, lbi, ubi = calc_bounds(man, element, nn, nlb, nub, relu_groups, destroy=False, use_krelu=refine)
        candidate_vars = [i for i, (l, u) in enumerate(zip(lbi, ubi)) if l<0 and u>0]

        if(refine):
            #use_milp = config.use_milp
            #if use_milp:
            #    timeout = timeout_milp
            #else:
            #    timeout = timeout_lp
            numconvslayers = sum('Conv2D' in l for l in nn.layertypes)
            #if numconvslayers-nn.conv_counter <= 1:

            #    resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, layerno, layerno, num_neurons, nlb, nub, use_milp,  candidate_vars, timeout)

            #    nlb[-1] = resl
            #    nub[-1] = resu

            #    for j in indices:
            #        update_bounds_for_neuron(man,element,layerno,j,resl[j],resu[j])

        elina_interval_array_free(bounds,num_neurons)
        nn.conv_counter+=1
        return element




class DeeppolyConv2dNodeFirst(DeeppolyConv2dNodeIntermediate):
    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        calc_bounds(man, element, nn, nlb, nub, relu_groups, use_krelu=refine)
        nn.conv_counter+=1
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


    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
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
        #calc_bounds(man, element, nn, nlb, nub)
        nn.maxpool_counter += 1
        return element


class DeeppolyResadd:
    def __init__(self, input_names, output_name, output_shape, has_relu):
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
        self.has_relu = has_relu
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
        if(self.has_relu):
             handle_residual_relu_layer(man,element,self.output_length,self.predecessors,use_area_heuristic)
        else:
             handle_residual_affine_layer(man,element,self.output_length,self.predecessors,use_area_heuristic)
        calc_bounds(man, element, nn, nlb, nub, relu_groups, use_krelu=refine)
        # print("Residual ", nn.layertypes[layerno],layerno)
        nn.residual_counter += 1

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

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
        handle_gather_layer(man, element, self.indexes)
        return element


class DeeppolySub:
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
        self.bias = np.ascontiguousarray(bias, dtype=np.uintp)
        self.is_minuend = is_minuend
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
        layerno = nn.calc_layerno()
        #handle_sub_layer(man, element, layerno, self.bias, self.is_minuend)
        nn.ffn_counter+=1
        return element


class DeeppolyMul:
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
        self.bias = np.ascontiguousarray(bias, dtype=np.uintp)
        add_input_output_information_deeppoly(self, input_names, output_name, output_shape)

    def transformer(self, nn, man, element, nlb, nub, relu_groups, refine, timeout_lp, timeout_milp, use_area_heuristic):
        layerno = nn.calc_layerno()
        #handle_mul_layer(man, element, layerno, self.bias)
        nn.ffn_counter+=1
        return element
