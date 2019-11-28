'''
@author: Adrian Hoffmann
'''
from deepzono_nodes import *
from deeppoly_nodes import *
from functools import reduce
import numpy as np


# TODO replace all strings with constants


class Optimizer:
    def __init__(self, operations, resources):
        """
        Arguments
        ---------
        operations : list
            list of dicts, each dict contains a mapping from a domain (like deepzono, refinezono or deeppoly) to a tuple with resources (like matrices, biases ...)
        resources : list
            list of str, each one being a type of operation (like "MatMul", "Conv2D", "Add" ...)
        """
        self.operations = operations
        self.resources  = resources
    
    
    def get_deepzono(self, nn, specLB, specUB = None):
        """
        This function will go through self.operations and self.resources and creates a list of Deepzono-Nodes which then can be run by an Analyzer object.
        It is assumed that self.resources[i]['deepzono'] holds the resources for the operation of type self.operations[i]                
        
        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec
        
        Return
        ------
        output : list
            list of Deepzono-Nodes that can be run by an Analyzer object
        """        
        output = []
        output_info = []
        domain = 'deepzono'
        nbr_op = len(self.operations)
        
        i = 0
        while i < nbr_op:
            output_info.append(self.resources[i][domain][-2:])
            if self.operations[i] == "Placeholder":
                input_names, output_name, output_shape = self.resources[i][domain]
                if specUB is None:
                    output.append(DeepzonoInputZonotope(specLB, input_names, output_name, output_shape))
                else:
                    output.append(DeepzonoInput(specLB, specUB, input_names, output_name, output_shape))
                i += 1
            elif self.operations[i] == "MatMul":
                if i != nbr_op-1 and self.operations[i+1] in ["Add", "BiasAdd"]:
                    matrix,  m_input_names, _, _           = self.resources[i][domain]
                    bias, _, output_name, b_output_shape = self.resources[i+1][domain]
                    
                    nn.weights.append(matrix)
                    nn.biases.append(bias)
                    nn.layertypes.append('Affine')
                    nn.numlayer+= 1
                    output.append(DeepzonoAffine(matrix, bias, m_input_names, output_name, b_output_shape))
                    i += 2
                else:
                    #self.resources[i][domain].append(refine)
                    output.append(DeepzonoMatmul(*self.resources[i][domain]))
                    i += 1
            elif self.operations[i] == "Gemm":
                matrix, bias, m_input_names, b_output_name, b_output_shape = self.resources[i][domain]

                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('Affine')
                nn.numlayer+= 1
                output.append(DeepzonoAffine(matrix, bias, m_input_names, b_output_name, b_output_shape))
                i += 1
            elif self.operations[i] == "Conv2D":
                if i != nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, c_input_names, _, _ = self.resources[i][domain]
                    bias, _, b_output_name, b_output_shape = self.resources[i+1][domain]
                    nn.numfilters.append(filters.shape[3])
                    nn.filter_size.append([filters.shape[0], filters.shape[1]])
                    nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                    nn.strides.append([strides[0],strides[1]])
                    nn.padding.append([pad_top, pad_left])
                    nn.out_shapes.append(b_output_shape)
                    nn.filters.append(filters)

                    nn.biases.append(bias)
                    nn.layertypes.append('Conv2D')
                    nn.numlayer+=1
                    output.append(DeepzonoConvbias(image_shape, filters, bias, strides, pad_top, pad_left, c_input_names, b_output_name, b_output_shape))
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, input_names, output_name, output_shape = self.resources[i][domain]
                    output.append(DeepzonoConv(image_shape, filters, strides, pad_top, pad_left, input_names, output_name, output_shape))
                    i += 1
            elif self.operations[i] == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, c_input_names, output_name, b_output_shape = self.resources[i][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(b_output_shape)
                nn.padding.append([pad_top, pad_left])
                nn.filters.append(filters)

                nn.biases.append(bias)
                nn.layertypes.append('Conv2D')
                nn.numlayer+=1
                output.append(DeepzonoConvbias(image_shape, filters, bias, strides, pad_top, pad_left, c_input_names, output_name, b_output_shape))
                i += 1
            elif self.operations[i] == "Add":
                #self.resources[i][domain].append(refine)
                output.append(DeepzonoAdd(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Sub":
                #self.resources[i][domain].append(refine)
                output.append(DeepzonoSub(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Mul":
                #self.resources[i][domain].append(refine)
                output.append(DeepzonoMul(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "MaxPool":
                image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape = self.resources[i][domain]
                output.append(DeepzonoMaxpool(image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape))
                i += 1
            elif self.operations[i] == "Resadd":
                #self.resources[i][domain].append(refine)
                output.append(DeepzonoResadd(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Relu":
                #self.resources[i][domain].append(refine)
                if nn.layertypes[-1]=='Affine':
                    nn.layertypes[-1] = 'ReLU'
                output.append(DeepzonoRelu(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Sigmoid":
                output.append(DeepzonoSigmoid(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Tanh":
                output.append(DeepzonoTanh(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Gather":
                image_shape, indexes, axis,  input_names, output_name, output_shape = self.resources[i][domain]
                calculated_indexes = self.get_gather_indexes(image_shape, indexes, axis)
                output.append(DeepzonoGather(calculated_indexes, input_names, output_name, output_shape))
                i += 1
            elif self.operations[i] == "Reshape":
                output.append(DeepzonoGather(*self.resources[i][domain]))
                i += 1
            else:
                assert 0, "the optimizer for Deepzono doesn't know of the operation type " + self.operations[i]

        use_dict = self.deepzono_get_dict(output)
        self.set_predecessors(nn, output)
        output   = self.deepzono_forward_pass(output, use_dict)

        return output, output_info


    def deepzono_get_dict(self, ir_list):
        """
        Returns a dict mapping output-names to the number of times that output is used by the nodes in the ir_list.
        This functions is a helper function for organizing the sections of an abstract elements when we have a ResNet or later an RNN.
        
        Arguments
        ---------
        ir_list : iterable
            list of Deepzono-Nodes
        
        Return
        ------
        use_dict : dict
            mapping from a name to the number of times that node-output is used
        """
        use_dict = {}
        for node in ir_list:
            for input_name in node.input_names:
                use_dict[input_name] += 1
            use_dict[node.output_name] = 0
        return use_dict


    def deepzono_forward_pass(self, ir_list, use_dict):
        """
        This function plans which Deepzono-Node-output occupies which section of an abstract element. If a DeepzonoDuplicate-Node should be needed, then this function will add it.
        This is needed when we have a ResNet or later RNNs.
        
        Arguments
        ---------
        ir_list : list
            list of Nodes, where each node has the fields output_length, input_names, and output_name (see DeepzonoNodes.py for examples)
        use_dict : dict 
            maps the output_name of each node in ir_list to the number of times the node's output will be used
        
        Return
        ------
        ir_list : list
            the ir_list with updated and potentionally added nodes
        """
        def get_index(active_abstracts, in_name, index_store):
            index = 0
            while True:
                index = index + active_abstracts[index:].index(in_name)
                if not index in index_store:
                    break
                index += 1
            return index


        active_abstracts = []
        abstract_length  = []

        i = 0
        while i < len(ir_list):
            node        = ir_list[i]
            index_store = []
            node.abstract_information = []
            for in_name in node.input_names:
                index  = get_index(active_abstracts, in_name, index_store)
                length = abstract_length[index]
                offset = reduce(lambda x,y: x+y, abstract_length[:index], 0)
                node.abstract_information += [offset, length]
                index_store.append(index)

            if len(index_store) != 0:
                active_abstracts[index_store[0]] = node.output_name
                abstract_length[index_store[0]]  = node.output_length
                for j in range(1,len(index_store)):
                    index = index_store[j]
                    del active_abstracts[index]
                    del abstract_length[index]
            else:
                active_abstracts.append(node.output_name)
                abstract_length.append(node.output_length)
                node.abstract_information = [0, node.output_length]

            i += 1

            if use_dict[node.output_name] > 1:
                for j in range(1, use_dict[node.output_name]):
                    ir_list.insert(i, DeepzonoDuplicate(node.abstract_information[0], node.output_length))
                    active_abstracts.append(node.output_name)
                    abstract_length.append(node.output_length)
                    i += 1

        return ir_list




    def get_deeppoly(self, nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size):
        """
        This function will go through self.operations and self.resources and create a list of Deeppoly-Nodes which then can be run by an Analyzer object.
        It is assumed that self.resources[i]['deeppoly'] holds the resources for an operation of type self.operations[i].
        self.operations should only contain a combination of the following 4 basic sequences:
            - Placholder         (only at the beginning)
                - MatMul -> Add -> Relu
                - Conv2D -> Add -> Relu    (not as last layer)
                - MaxPool         (only as intermediate layer)    
        
        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec
        
        Return
        ------
        output : list
            list of Deeppoly-Nodes that can be run by an Analyzer object
        """
        output = []
        output_info = []
        domain = 'deeppoly'

        i = 0
        while i < len(self.operations):
            output_info.append(self.resources[i][domain][-2:])
            #print(self.operations[i])
            if self.operations[i] == "Placeholder":
                input_names, output_name, output_shape = self.resources[i][domain]
                output.append(DeeppolyInput(specLB, specUB, input_names, output_name, output_shape,
                                            lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size))
                i += 1

            # Tensorflow operation
            elif self.operations[i] == "MatMul" and self.operations[i+1] in ["Add", "BiasAdd"]:
                matrix, input_names, _,_ = self.resources[i][domain]
                bias,_, output_name, output_shape = self.resources[i+1][domain]
                if i != len(self.operations) - 2 and self.operations[i + 2] in ["Relu", "Sigmoid", "Tanh"]:
                    _,output_name, output_shape = self.resources[i+2][domain]
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('Affine')
                nn.numlayer+= 1

                if i == 1:
                    if (self.operations[i + 2] == "Relu"):
                        nn.layertypes[-1] = 'ReLU'
                        output.append(DeeppolyReluNodeFirst(matrix, bias, input_names, output_name, output_shape))
                    elif (self.operations[i + 2] == "Sigmoid"):
                        output.append(DeeppolySigmoidNodeFirst(matrix, bias, input_names, output_name, output_shape))
                    elif (self.operations[i + 2] == "Tanh"):
                        output.append(DeeppolyTanhNodeFirst(matrix, bias, input_names, output_name, output_shape))
                    else:
                        assert 0
                    i += 3

                elif i == len(self.operations) - 2:
                    output.append(DeeppolyReluNodeLast(matrix, bias, False, input_names, output_name, output_shape))
                    i += 2

                elif i == len(self.operations) - 3:
                    if(self.operations[i+2] == "Relu"):
                        nn.layertypes[-1] = 'ReLU'
                        output.append(DeeppolyReluNodeLast(matrix, bias, True, input_names, output_name, output_shape))
                    elif(self.operations[i+2] == "Sigmoid"):
                        output.append(DeeppolySigmoidNodeLast(matrix, bias, True, input_names, output_name, output_shape))
                    elif(self.operations[i+2] == "Tanh"):
                        output.append(DeeppolyTanhNodeLast(matrix, bias, True, input_names, output_name, output_shape))
                    else:
                        assert 0, self.operations[i+2]
                    i += 3

                else:
                    if(self.operations[i+2] == "Relu"):
                        nn.layertypes[-1] = 'ReLU'
                        output.append(DeeppolyReluNodeIntermediate(matrix, bias, input_names, output_name, output_shape))
                    elif(self.operations[i+2] == "Sigmoid"):
                        output.append(DeeppolySigmoidNodeIntermediate(matrix, bias, input_names, output_name, output_shape))
                    elif(self.operations[i+2]=="Tanh"):
                        output.append(DeeppolyTanhNodeIntermediate(matrix, bias, input_names, output_name, output_shape))
                    else:
                        assert 0
                    i += 3


            # ONNX operation
            elif self.operations[i] == "Gemm" :
                matrix, bias, input_names,output_name,output_shape = self.resources[i][domain]
                if i != len(self.operations) - 1:
                    _,output_name, output_shape = self.resources[i+1][domain]
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('Affine')
                nn.numlayer+= 1

                if i == 1:
                    if (self.operations[i + 1] == "Relu"):
                        nn.layertypes[-1] = 'ReLU'
                        output.append(DeeppolyReluNodeFirst(matrix, bias, input_names, output_name, output_shape))
                    elif (self.operations[i + 1] == "Sigmoid"):
                        output.append(DeeppolySigmoidNodeFirst(matrix, bias, input_names, output_name, output_shape))
                    elif (self.operations[i + 1] == "Tanh"):
                        output.append(DeeppolyTanhNodeFirst(matrix, bias, input_names, output_name, output_shape))
                    else:
                        assert 0
                    i += 2

                elif i == len(self.operations) - 2:
                    if(self.operations[i+1] == "Relu"):
                        nn.layertypes[-1] = 'ReLU'
                        output.append(DeeppolyReluNodeLast(matrix, bias, True, input_names, output_name, output_shape))
                    elif(self.operations[i+1] == "Sigmoid"):
                        output.append(DeeppolySigmoidNodeLast(matrix, bias, True, input_names, output_name, output_shape))
                    elif(self.operations[i+1] == "Tanh"):
                        output.append(DeeppolyTanhNodeLast(matrix, bias, True, input_names, output_name, output_shape))
                    else:
                        assert 0
                    i += 2

                elif i == len(self.operations) - 1:
                    output.append(DeeppolyReluNodeLast(matrix, bias, False, input_names, output_name, output_shape))
                    i += 1

                else:
                    if(self.operations[i+1] == "Relu"):
                        nn.layertypes[-1] = 'ReLU'
                        output.append(DeeppolyReluNodeIntermediate(matrix, bias, input_names, output_name, output_shape))
                    elif(self.operations[i+1] == "Sigmoid"):
                        output.append(DeeppolySigmoidNodeIntermediate(matrix, bias, input_names, output_name, output_shape))
                    elif(self.operations[i+1]=="Tanh"):
                        output.append(DeeppolyTanhNodeIntermediate(matrix, bias, input_names, output_name, output_shape))
                    else:
                        assert 0
                    i += 2

            elif self.operations[i] == "MaxPool":
                image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape = self.resources[i][domain]
                output.append(DeeppolyMaxpool(image_shape, window_size, strides, input_names, output_name, output_shape))
                i += 1

                # Tensorflow operation
            elif self.operations[i] == "Conv2D" and self.operations[i + 1] == "BiasAdd":

                filters, image_shape, strides, pad_top, pad_left, input_names, _, _ = self.resources[i][domain]
                bias, _, output_name, output_shape = self.resources[i + 1][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0], image_shape[1], image_shape[2]])
                nn.strides.append([strides[0], strides[1]])
                nn.padding.append([pad_top, pad_left])
                nn.out_shapes.append(output_shape)
                nn.filters.append(filters)

                nn.biases.append(bias)
                has_relu = self.operations[i + 2] == "Relu"
                if has_relu:
                    nn.layertypes.append('Conv2D')
                    _, output_name, output_shape = self.resources[i + 2][domain]
                    increment = 3
                else:
                    nn.layertypes.append('Conv2DNoReLU')
                    increment = 2
                nn.numlayer += 1
                if i == 1:
                    output.append(
                        DeeppolyConv2dNodeFirst(filters, strides, pad_top, pad_left, bias, image_shape, input_names, output_name,
                                                output_shape, has_relu))
                else:
                    output.append(DeeppolyConv2dNodeIntermediate(filters, strides, pad_top, pad_left, bias, image_shape, input_names,
                                                             output_name, output_shape, has_relu))
                i += increment

            # ONNX operation
            elif self.operations[i] == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, input_names,output_name,out_shape = self.resources[i][domain]
                has_relu = self.operations[i+1] == "Relu"
                if has_relu:
                    nn.layertypes.append('Conv2D')
                    _,output_name,output_shape = self.resources[i+1][domain]
                    increment = 2
                else:
                    nn.layertypes.append('Conv2DNoReLU')
                    increment = 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.padding.append([pad_top, pad_left])
                nn.out_shapes.append(out_shape)
                nn.filters.append(filters)

                nn.biases.append(bias)
                nn.numlayer+=1
                if i == 1:
                    output.append(DeeppolyConv2dNodeFirst(filters, strides, pad_top, pad_left, bias, image_shape, input_names, output_name, output_shape, has_relu))
                else:
                    output.append(DeeppolyConv2dNodeIntermediate(filters, strides, pad_top, pad_left, bias, image_shape, input_names, output_name, output_shape, has_relu))
                i += increment

            # Residual layer
            elif self.operations[i] == "Resadd":
                if self.operations[i+1] == "Relu":
                    input_names,_,_ = self.resources[i][domain]
                    _,output_name,output_shape = self.resources[i+1][domain]
                    output.append(DeeppolyResadd(input_names,output_name,output_shape, True))
                    nn.layertypes.append('Resadd')
                    i += 2
                else:
                    input_names,output_name,output_shape = self.resources[i][domain]
                    output.append(DeeppolyResadd(input_names,output_name,output_shape, False))
                    nn.layertypes.append('Resaddnorelu')
                    i += 1
                nn.numlayer+=1

            # Gather
            elif self.operations[i] == "Gather":
                image_shape, indexes, axis, input_names,output_name,output_shape = self.resources[i][domain]
                calculated_indexes = self.get_gather_indexes(image_shape, indexes, axis)
                output.append(DeeppolyGather(calculated_indexes, input_names,output_name,output_shape))
                i+=1

            elif self.operations[i] == "Reshape":
                indexes, input_names, output_name, output_shape = self.resources[i][domain]
                output.append(DeeppolyGather(indexes, [input_names[0]], output_name, output_shape))
                i+=1

            elif self.operations[i] == "Sub":
                output.append(DeeppolySub(*self.resources[i][domain]))
                i += 1

            elif self.operations[i] == "Mul":
                output.append(DeeppolyMul(*self.resources[i][domain]))
                i += 1
            else:
                assert 0, "the Deeppoly analyzer doesn't support the operation: '" + self.operations[i] + "' of this network"

        self.set_predecessors(nn, output)
        return output, output_info

    def set_predecessors(self, nn, output):
        output_index_store = {}
        index_o = 0
        for node in output:
            output_index_store[node.output_name] = index_o
            index_o += 1
        for node in output:
            predecessors = (c_size_t * len(node.input_names))()
            i = 0
            for input_name in node.input_names:
                predecessors[i] = output_index_store[input_name]
                i += 1
            node.predecessors = predecessors
            nn.predecessors.append(predecessors)

    def get_gather_indexes(self, input_shape, indexes, axis):
        size = np.prod(input_shape)
        base_indexes = np.arange(size).reshape(input_shape)
        return np.take(base_indexes, indexes, axis=axis)