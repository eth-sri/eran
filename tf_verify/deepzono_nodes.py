"""
@author: Adrian Hoffmann
"""


import numpy as np
from zonoml import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from ctypes.util import find_library
from functools import reduce
from deepzono_milp import *



def add_dimensions(man, element, offset, n):
    """
    adds dimensions to an abstract element
    
    Arguments
    ---------
    man : ElinaManagerPtr
        manager which is responsible for element
    element : ElinaAbstract0Ptr
        the element to which dimensions get added
    offset : int
        offset at which the dimensions should be added
    n : int
        n dimensions will be added to element at offset
    
    Return
    ------
    output : ElinaAbstract0Ptr
        new abstract element with the added dimensions
    """
    dimchange_ptr = elina_dimchange_alloc(0, n)
    elina_dimchange_init(dimchange_ptr, 0, n)
    for i in range(n):
        dimchange_ptr.contents.dim[i] = offset
    output = elina_abstract0_add_dimensions(man, True, element, dimchange_ptr, False)
    elina_dimchange_free(dimchange_ptr)
    return output


def remove_dimensions(man, element, offset, n):
    """
    removes dimensions from an abstract element
    
    Arguments
    ---------
    man : ElinaManagerPtr
        manager which is responsible for element
    element : ElinaAbstract0Ptr
        the element from which dimensions get removed
    offset : int
        offset form which on the dimensions should be removed
    n : int
        n dimensions will be removed from the element at offset
    
    Return
    ------
    output : ElinaAbstract0Ptr
        new abstract element with the n dimensions removed
    """
    dimchange_ptr = elina_dimchange_alloc(0, n)
    elina_dimchange_init(dimchange_ptr, 0, n)
    for i in range(n):
        dimchange_ptr.contents.dim[i] = offset+i
    output = elina_abstract0_remove_dimensions(man, True, element, dimchange_ptr)
    elina_dimchange_free(dimchange_ptr)
    return output


def get_xpp(matrix):
    """
    Arguments
    ---------
    matrix : numpy.ndarray
        must be a 2D array
    
    Return
    ------
    output : numpy.ndarray
        contains pointers to the rows of matrix
    """
    return (matrix.__array_interface__['data'][0]+ np.arange(matrix.shape[0])*matrix.strides[0]).astype(np.uintp)


def add_input_output_information(self, input_names, output_name, output_shape):
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



def refine_relu_with_solver_bounds(nn, self, man, element, nlb, nub, timeout_lp, timeout_milp):
    """
    refines the relu transformer

    Arguments
    ---------
    self : Object
        will be a DeepzonoNode, but could be any object
    man : ElinaManagerPtr
        manager which is responsible for element
    element : ElinaAbstract0Ptr
        the element in which the results after affine transformation are stored
    nlb: list of list of doubles
        contains the lower bounds for all neurons upto layer layerno
    nub: list of list of doubles
        contains the upper bounds for all neurons upto layer layerno
    use_milp: bool array
        whether to use milp or lp for refinement
    Return
    ------
     the updated abstract element
    """
    
    offset, length = self.abstract_information
    layerno = nn.ffn_counter + nn.conv_counter
    if layerno==0 or nn.last_layer=='Conv2D':
        element = relu_zono_layerwise(man,True,element,offset, length) 
    else:
        is_conv = False
        timeout = timeout_milp
        for i in range(nn.numlayer):
            if nn.layertypes[i] == 'Conv2D':
                is_conv = True
                break
        use_milp = 0
        if is_conv==True:
            use_milp = 1
        else:
            if layerno<=3:
               use_milp = 1
            else:
               use_milp = 0
               timeout = timeout_lp
        lbi = nlb[layerno]
        ubi = nub[layerno]
        candidate_vars = []
        for i in range(length):
            if((lbi[i]<0 and ubi[i]>0) or (lbi[i]>0)):
                 candidate_vars.append(i)
        #TODO handle residual layers here
        resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, layerno, layerno, length, nlb, nub, use_milp,  candidate_vars, timeout)
        nlb.pop()
        nub.pop()
        nlb.append(resl)
        nub.append(resu)

        j = 0
     
        for i in range(length):
            if((j < len(indices)) and (i==indices[j])):
             
                element = relu_zono_refined(man,True, element,i+offset, resl[i],resu[i])
                j=j+1
            else:
                element = relu_zono(man,True,element,i+offset)
      
     
    return element

class DeepzonoInput:
    def __init__(self, specLB, specUB, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information(self, input_names, output_name, output_shape)
        self.specLB = np.ascontiguousarray(specLB, dtype=np.double)
        self.specUB = np.ascontiguousarray(specUB, dtype=np.double)
    
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
        return zonotope_from_network_input(man, 0, len(self.specLB), self.specLB, self.specUB)




class DeepzonoMatmul:
    def __init__(self, matrix, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        matrix : numpy.ndarray
            2D matrix for the matrix multiplication
        input_names : iterable
            iterable with the name of the vector for the matrix multiplication
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information(self, input_names, output_name, output_shape)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.double)
        #self.refine = refine
    
    
    def get_arguments(self, man, element):
        """
        used to get the arguments to the transformer, also used by the child class
        Note: this function also adds the necessary dimensions, removing the old ones after the transformer is the responsibility of the caller
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : tuple
            arguments to conv_matmult_zono, see zonoml.py for more information
        """
        offset, old_length = self.abstract_information
        new_length         = self.output_length
        element            = add_dimensions(man, element, offset + old_length, new_length)
        matrix_xpp         = get_xpp(self.matrix)
        return man, True, element, offset+old_length, matrix_xpp, new_length, offset, old_length
    
    
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with ffn_matmult_without_bias_zono
        
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
        offset, old_length = self.abstract_information
        element = ffn_matmult_without_bias_zono(*self.get_arguments(man, element))
        #if self.refine == 'True':
        #    refine_after_affine(self, man, element, nlb, nub)

        return remove_dimensions(man, element, offset, old_length)





class DeepzonoAdd:
    def __init__(self, bias, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        bias : numpy.ndarray
            the values of the first addend
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information(self, input_names, output_name, output_shape)
        self.bias = np.ascontiguousarray(bias, dtype=np.double)
    
    
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with ffn_add_bias_zono
        
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
        offset, old_length = self.abstract_information
        return ffn_add_bias_zono(man, True, element, offset, self.bias, old_length)





class DeepzonoAffine(DeepzonoMatmul):
    def __init__(self, matrix, bias, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        matrix : numpy.ndarray
            2D matrix for the matrix multiplication
        bias : numpy.ndarray
            the values of the bias
        input_names : iterable
            iterable with the name of the other addend of the addition
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        DeepzonoMatmul.__init__(self, matrix, input_names, output_name, output_shape)
        self.bias = np.ascontiguousarray(bias, dtype=np.double)
        #self.refine = refine    

    
    def transformer(self, nn, man, element,nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with ffn_matmult_zono
        
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
        offset, old_length = self.abstract_information
        man, destructive, element, start_offset, weights, num_vars, expr_offset, expr_size = self.get_arguments(man, element)
        element = ffn_matmult_zono(man, destructive, element, start_offset, weights, self.bias, num_vars, expr_offset, expr_size)
        #if self.refine == 'True':
        #    refine_after_affine(self, man, element, nlb, nub)
        dimension = elina_abstract0_dimension(man,element)
        var_in_element = dimension.intdim + dimension.realdim 
        bounds = elina_abstract0_to_box(man,element)
        lbi = []
        ubi = []
        for i in range(num_vars):
            inf = bounds[i+start_offset].contents.inf
            sup = bounds[i+start_offset].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 

        elina_interval_array_free(bounds,var_in_element)
        nn.last_layer = 'Affine'
        return remove_dimensions(man, element, offset, old_length)
        



class DeepzonoConv:
    def __init__(self, image_shape, filters, strides, padding, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        image_shape : numpy.ndarray
            of shape [height, width, channels]
        filters : numpy.ndarray
            the 4D array with the filter weights
        strides : numpy.ndarray
            of shape [height, width]
        padding : str
            type of padding, either 'VALID' or 'SAME'
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information(self, input_names, output_name, output_shape)
        self.image_size = np.ascontiguousarray(image_shape, dtype=np.uintp)
        self.filters    = np.ascontiguousarray(filters, dtype=np.double)
        self.strides    = np.ascontiguousarray(strides, dtype=np.uintp)
        self.padding    = padding
    
    
    def get_arguments(self, man, element):
        """
        used to get the arguments to the transformer, also used by the child class
        Note: this function also adds the necessary dimensions, removing the old ones after the transformer is the responsibility of the caller
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : tuple
            arguments to conv_matmult_zono, see zonoml.py for more information
        """
        offset, old_length  = self.abstract_information
        filter_size = (c_size_t * 2) (self.filters.shape[0], self.filters.shape[1])
        num_filters = self.filters.shape[3]
        new_length  = self.output_length
        image_size  = (c_size_t * 3)(self.image_size[0],self.image_size[1],self.image_size[2])
        strides     = (c_size_t * 2)(self.strides[0], self.strides[1])
        element     = add_dimensions(man, element, offset+old_length, new_length)
        return man, True, element, old_length+offset, self.filters, np.ndarray([0,0,0]), image_size, offset, filter_size, num_filters, strides, self.padding == "VALID", False
        
    
    
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with conv_matmult_zono, without bias
        
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
        offset, old_length  = self.abstract_information
        element = conv_matmult_zono(*self.get_arguments(man, element))
        
        
        return remove_dimensions(man, element, offset, old_length)



class DeepzonoConvbias(DeepzonoConv):
    def __init__(self, image_shape, filters, bias, strides, padding, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        image_shape : numpy.ndarray
            of shape [height, width, channels]
        filters : numpy.ndarray
            the 4D array with the filter weights
        bias : numpy.ndarray
            array with the bias (has to have as many elements as the filter has out channels)
        strides : numpy.ndarray
            of shape [height, width]
        padding : str
            type of padding, either 'VALID' or 'SAME'
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        DeepzonoConv.__init__(self, image_shape, filters, strides, padding, input_names, output_name, output_shape)
        self.bias = np.ascontiguousarray(bias, dtype=np.double)
    
    
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with conv_matmult_zono, with bias
        
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
        offset, old_length  = self.abstract_information
        man, destructive, element, start_offset, filters, bias, input_size, expr_offset, filter_size, num_filters, strides, is_valid_padding, has_bias = self.get_arguments(man, element)
        bias     = self.bias
        has_bias = True
        #print("start ", start_offset,expr_offset,filter_size,num_filters)
        element = conv_matmult_zono(man, destructive, element, start_offset, filters, bias, input_size, expr_offset, filter_size, num_filters, strides, is_valid_padding, has_bias)
        num_vars = self.output_length
        #print("coming here")
        dimension = elina_abstract0_dimension(man,element)
        num_neurons = dimension.intdim + dimension.realdim
        bounds = elina_abstract0_to_box(man,element)
        lbi = []
        ubi = []
        for i in range(num_vars):
            inf = bounds[i+start_offset].contents.inf
            sup = bounds[i+start_offset].contents.sup
            lbi.append(inf.contents.val.dbl)
            ubi.append(sup.contents.val.dbl)
    
        nlb.append(lbi)
        nub.append(ubi) 

        elina_interval_array_free(bounds,num_neurons)
        nn.last_layer='Conv2D'
        return remove_dimensions(man, element, offset, old_length)




class DeepzonoNonlinearity:
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
        add_input_output_information(self, input_names, output_name, output_shape)
    
    
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
        offset, length = self.abstract_information
        return man, True, element, offset, length




class DeepzonoRelu(DeepzonoNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with relu_zono_layerwise
        
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
        if refine==True:
            element = refine_relu_with_solver_bounds(nn, self, man, element, nlb, nub, timeout_lp, timeout_milp)
        else:
            element = relu_zono_layerwise(*self.get_arguments(man, element))

        if nn.last_layer=='Affine':
           nn.ffn_counter+=1
        elif nn.last_layer == 'Conv2D':
           nn.conv_counter+=1
        return element




class DeepzonoSigmoid(DeepzonoNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with sigmoid_zono_layerwise
        
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
        return sigmoid_zono_layerwise(*self.get_arguments(man, element))




class DeepzonoTanh(DeepzonoNonlinearity):
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with tanh_zono_layerwise
        
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
        return tanh_zono_layerwise(*self.get_arguments(man, element))




class DeepzonoMaxpool:
    def __init__(self, image_shape, window_size, strides, padding, input_names, output_name, output_shape):
        """
        Arguments
        ---------
        image_shape : numpy.ndarray
            1D array of shape [height, width, channels]
        window_size : numpy.ndarray
            1D array of shape [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of shape [height, width] representing the stride in these directions
        padding : str
            type of padding, either 'VALID' or 'SAME'
        input_names : iterable
            iterable with the name of node output we apply maxpool on
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information(self, input_names, output_name, output_shape)
        self.window_size = np.ascontiguousarray(window_size, dtype=np.uintp)
        self.input_shape = np.ascontiguousarray(image_shape, dtype=np.uintp)
        self.stride      = np.ascontiguousarray(strides, dtype=np.uintp)
        self.padding     = padding
        
    
    
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        transforms element with maxpool_zono
        
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
        offset, old_length = self.abstract_information
        h, w    = self.window_size
        H, W, C = self.input_shape
        element = maxpool_zono(man, True, element, (c_size_t * 3)(h,w,1), (c_size_t * 3)(H, W, C), 0, (c_size_t * 2)(self.stride[0], self.stride[1]), 3, offset+old_length, self.padding=="VALID")
        return remove_dimensions(man, element, offset, old_length)




class DeepzonoDuplicate:
    def __init__(self, src_offset, num_var):
        """
        Arguments
        ---------
        src_offset : int
            the section that need to be copied starts at src_offset
        num_var : int
            how many dimensions should be copied
        """
        self.src_offset = src_offset
        self.num_var    = num_var
        
        
    def transformer(self, nn, man, element, nlb, nub, refine, timeout_lp, timeout_milp):
        """
        adds self.num_var dimensions to element and then fills these dimensions with zono_copy_section
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            element with the specified section copied to the end
        """
        dst_offset = elina_abstract0_dimension(man, element).realdim
        add_dimensions(man, element, dst_offset, self.num_var)
        zono_copy_section(man, element, dst_offset, self.src_offset, self.num_var)
        return element




class DeepzonoResadd:
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
        add_input_output_information(self, input_names, output_name, output_shape)
        
    
    def transformer(self, nn, man, element, nlb,nub, refine, timeout_lp, timeout_milp):
        """
        uses zono_add to add two sections from element together and removes the section that is defined by self.abstract_information[2]
        the result of the addition is stored in the section defined by self.abstract_information[:2]
        
        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            resulting element
        """
        dst_offset, num_var = self.abstract_information[:2]
        src_offset = self.abstract_information[2]
        zono_add(man, element, dst_offset, src_offset, num_var)
        if dst_offset == src_offset:
            return element
        else:
            return remove_dimensions(man, element, src_offset, num_var)
