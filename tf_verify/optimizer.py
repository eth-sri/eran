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

import warnings
from deepzono_nodes import *
from deeppoly_nodes import *
# if config.domain=='gpupoly' or config.domain=='refinegpupoly':
try:
    from gpupoly import Network
    GPU_FLAG = True
except:
    GPU_FLAG = False
    warnings.warn("gpupoly not available.")
from functools import reduce
import numpy as np
from read_net_file import *


operations_for_neuron_count = ["Relu", "Sigmoid", "Tanh", "MaxPool", "LeakyRelu"]


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

    def get_neuron_count(self):
        total_neurons = 0
        for op, res in zip(self.operations, self.resources):
            if op in operations_for_neuron_count:
                if len(res['deepzono'][-1])==4:
                    total_neurons += np.prod(res['deepzono'][-1][1:len(res['deepzono'][-1])])
                else:
                    total_neurons += np.prod(res['deepzono'][-1][0:len(res['deepzono'][-1])])
        return total_neurons

    def get_abstract_element(self, nn, i, execute_list, output_info, domain):
        assert domain == "deepzono" or domain == "deeppoly", "ERAN does not support" + domain + " abstraction"
        nbr_op = len(self.operations)
        while i < nbr_op:
            if self.operations[i] == "MatMul":
                nn.layertypes.append('FC')
                if i < nbr_op-1 and self.operations[i+1] in ["Add", "BiasAdd"]:
                    matrix,  m_input_names, _, _           = self.resources[i][domain]
                    bias, _, output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    #self.resources[i][domain].append(refine)
                    matrix, m_input_names , output_name , b_output_shape  = self.resources[i][domain]
                    
                    bias_length = reduce((lambda x, y: x*y), b_output_shape)
                    bias = np.zeros(bias_length)

                    i += 1
                if domain == 'deepzono':
                    execute_list.append(DeepzonoAffine(matrix, bias, m_input_names, output_name, b_output_shape))
                elif domain == 'deeppoly':
                    execute_list.append(DeeppolyFCNode(matrix, bias, m_input_names, output_name, b_output_shape))
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.numlayer+= 1
            elif self.operations[i] == "Gemm":
                matrix, bias, m_input_names, b_output_name, b_output_shape = self.resources[i][domain]

                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('FC')
                nn.numlayer+= 1
                #print("Gemm Matrix ", matrix)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoAffine(matrix, bias, m_input_names, b_output_name, b_output_shape))
                else:
                    execute_list.append(DeeppolyFCNode(matrix, bias, m_input_names, b_output_name, b_output_shape))
                i += 1
            
            elif self.operations[i] == "Conv2D":
                if i < nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, _, _ = self.resources[i][domain]
                    bias, _, b_output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, b_output_name, b_output_shape = self.resources[i][domain]
                    bias_length = reduce((lambda x, y: x*y), output_shape)
                    bias = np.zeros(bias_length)
                    i += 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.out_shapes.append(b_output_shape)
                nn.filters.append(filters)
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                if domain == 'deepzono':
                    execute_list.append(DeepzonoConvbias(image_shape, filters, bias, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, b_output_name, b_output_shape))
                else:
                    execute_list.append(DeeppolyConv2dNode(filters, strides, pad_top, pad_left, pad_bottom, pad_right, bias, image_shape, c_input_names, b_output_name, b_output_shape))
                nn.numlayer+=1
            elif self.operations[i] == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, output_name, b_output_shape = self.resources[i][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(b_output_shape)
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.filters.append(filters)

                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                nn.numlayer+=1
                if domain == 'deepzono':
                    execute_list.append(DeepzonoConvbias(image_shape, filters, bias, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, output_name, b_output_shape))
                else:
                    execute_list.append(DeeppolyConv2dNode(filters, strides, pad_top, pad_left, pad_bottom, pad_right, bias, image_shape, c_input_names, output_name, b_output_shape))
                i += 1    
            elif self.operations[i] == "Resadd":
                #self.resources[i][domain].append(refine)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoResadd(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolyResidualNode(*self.resources[i][domain]))
                nn.layertypes.append('Resadd')
                nn.numlayer += 1
                i += 1
            #elif self.operations[i] == "Add":
                #self.resources[i][domain].append(refine)
           #     execute_list.append(DeepzonoAdd(*self.resources[i][domain]))
           #     nn.layertypes.append('Add')
           #     nn.numlayer += 1
           #     i += 1
            elif self.operations[i] == "Sub":
                #self.resources[i][domain].append(refine)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoSub(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolySubNode(*self.resources[i][domain]))
                nn.layertypes.append('Sub')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Mul":
                #self.resources[i][domain].append(refine)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoMul(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolyMulNode(*self.resources[i][domain]))
                nn.layertypes.append('Mul')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "MaxPool" or self.operations[i] == "AveragePool" or self.operations[i] == "AvgPool":
                image_shape, window_size, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, output_shape = self.resources[i][domain]
                nn.pool_size.append(window_size)
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(output_shape)
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.numlayer+=1
                is_maxpool = (self.operations[i]=="MaxPool")
                if is_maxpool:
                    nn.layertypes.append('Maxpool')
                else:
                    nn.layertypes.append('Avgpool')
                if domain == 'deepzono':
                    execute_list.append(DeepzonoPool(image_shape, window_size, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, output_shape, is_maxpool))
                else:
                    execute_list.append(DeeppolyPoolNode(image_shape, window_size, strides,pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, output_shape, is_maxpool))
                i += 1
            elif self.operations[i] == "Pad":
                image_shape, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, output_shape = self.resources[i][domain]
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.out_shapes.append(output_shape)
                nn.layertypes.append('Pad')
                if domain == 'deepzono':
                    #execute_list.append(DeepzonoPad(image_shape, filters, bias, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, b_output_name, b_output_shape))
                    raise NotImplementedError
                else:
                    execute_list.append(DeeppolyPaddingNode(pad_top, pad_left, pad_bottom, pad_right, image_shape, input_names, output_name, output_shape))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Relu":
                #self.resources[i][domain].append(refine)
                nn.layertypes.append('ReLU')
                if domain == 'deepzono':
                    execute_list.append(DeepzonoRelu(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolyReluNode(*self.resources[i][domain]))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Sign":
                nn.layertypes.append('Sign')
                if domain == 'deeppoly':
                   execute_list.append(DeeppolySignNode(*self.resources[i][domain]))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "LeakyRelu":
                nn.layertypes.append('LeakyRelu')
                if domain == 'deeppoly':
                   execute_list.append(DeeppolyLeakyReluNode(*self.resources[i][domain]))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Sigmoid":
                if domain == 'deepzono':
                    execute_list.append(DeepzonoSigmoid(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolySigmoidNode(*self.resources[i][domain]))
                nn.layertypes.append('Sigmoid')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Tanh":
                if domain == 'deepzono':
                    execute_list.append(DeepzonoTanh(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolyTanhNode(*self.resources[i][domain]))
                nn.layertypes.append('Tanh')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Gather":
                image_shape, indexes, axis,  input_names, output_name, output_shape = self.resources[i][domain]
                calculated_indexes = self.get_gather_indexes(image_shape, indexes, axis)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoGather(calculated_indexes, input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyGather(calculated_indexes, input_names, output_name, output_shape))
                nn.layertypes.append('Gather')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Reshape":
                indexes, input_names, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoGather(*self.resources[i][domain]))
                else:
                    execute_list.append(DeeppolyGather(indexes, [input_names[0]], output_name, output_shape))
                nn.layertypes.append('Gather')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Concat":
                assert domain == "deeppoly", "Only DeepPoly currently supports concatenation"
                width, height, channels, input_names, output_name, output_shape = self.resources[i][domain]
                execute_list.append(DeeppolyConcat(width, height, channels, input_names, output_name, output_shape))
                nn.layertypes.append('Concat')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Tile":
                assert domain == "deeppoly", "Only DeepPoly currently supports tiling"
                repeats, input_names, output_name, output_shape = self.resources[i][domain]
                execute_list.append(DeeppolyTile(repeats, input_names, output_name, output_shape))
                nn.layertypes.append('Tile')
                nn.numlayer += 1
                i += 1
            else:
                assert 0, "the optimizer for" + domain + " doesn't know of the operation type " + self.operations[i]
            output_info.append(self.resources[i-1][domain][-2:])

                
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
        execute_list : list
            list of Deepzono-Nodes that can be run by an Analyzer object
        """        
        execute_list = []
        output_info = []
        domain = 'deepzono'
        nbr_op = len(self.operations)
        
        
        assert self.operations[0] == "Placeholder", "the optimizer for Deepzono cannot handle this network "
        input_names, output_name, output_shape = self.resources[0][domain]
        if specUB is None:
            execute_list.append(DeepzonoInputZonotope(specLB, input_names, output_name, output_shape))
        else:
            execute_list.append(DeepzonoInput(specLB, specUB, input_names, output_name, output_shape))
        output_info.append(self.resources[0][domain][-2:])

                
        self.get_abstract_element(nn, 1, execute_list, output_info, 'deepzono')


        # for testing, getting the corresponding layer in the tensorflow or onnx model

        use_dict = self.deepzono_get_dict(execute_list)
        self.set_predecessors(nn, execute_list)
        execute_list   = self.deepzono_forward_pass(execute_list, use_dict)
        return execute_list, output_info


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




    def get_deeppoly(self, nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints=None):
        """
        This function will go through self.operations and self.resources and create a list of Deeppoly-Nodes which then can be run by an Analyzer object.
        It is assumed that self.resources[i]['deeppoly'] holds the resources for an operation of type self.operations[i].
        self.operations should only contain a combination of the following 4 basic sequences:
            - Placholder         (only at the beginning)
                - MatMul -> Add -> Relu
                - Conv2D -> Add -> Relu    (not as last layer)
                - MaxPool/AveragePool         (only as intermediate layer)

        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec

        Return
        ------
        execute_list : list
            list of Deeppoly-Nodes that can be run by an Analyzer object
        """
        execute_list = []
        output_info = []
        domain = 'deeppoly'
        assert self.operations[0] == "Placeholder", "the optimizer for Deeppoly cannot handle this network "
        input_names, output_name, output_shape = self.resources[0][domain]
        output_info.append(self.resources[0][domain][-2:])
        execute_list.append(DeeppolyInput(specLB, specUB, input_names, output_name, output_shape,
                                            lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints))

        self.get_abstract_element(nn, 1, execute_list, output_info, 'deeppoly')
        self.set_predecessors(nn, execute_list)
        return execute_list, output_info

    def get_gpupoly(self, nn):
        assert GPU_FLAG, "GPUPoly is not available"
        domain = 'deeppoly'
        input_names, output_name, output_shape = self.resources[0][domain]
        #print("output ", np.prod(output_shape))
        network = Network(np.prod(output_shape))
        nbr_op = len(self.operations)
        i=1
        num_gpu_layers = 1
        relu_layers = []
        last_layer = None
        while i < nbr_op:
            #TODO support Maxpool
            if self.operations[i] == "MatMul":
                nn.layertypes.append('FC')
                
                if i < nbr_op-1 and self.operations[i+1] in ["Add", "BiasAdd"]:
                    matrix,  m_input_names, _, _           = self.resources[i][domain]
                    bias, _, output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    #self.resources[i][domain].append(refine)
                    matrix, m_input_names , output_name , b_output_shape  = self.resources[i][domain]
                    
                    bias_length = reduce((lambda x, y: x*y), b_output_shape)
                    bias = np.zeros(bias_length)
                    i += 1
                #if last_layer=="Conv":
                    
                #    output_shape = nn.out_shapes[-1]
                #    h,w,c = [output_shape[1], output_shape[2],output_shape[3]]
                #    new_matrix = permutation(matrix, h, w, c)
                    
                    #num_var = len(matrix)
                    #new_matrix = np.zeros((num_var, np.prod(output_shape)),dtype=np.double)
                    #new_shape = [output_shape[3],output_shape[1], output_shape[2]]
                    #print("coming here", np.prod(output_shape), output_shape)
                    #for var in range(num_var):
                    #    for h in range(output_shape[1]):
                    #        for w in range(output_shape[2]):
                    #            for c in range(output_shape[3]):
                    #                print("HWC ", h,w,c, c*new_shape[1] * new_shape[2] + h* new_shape[2] + w,h*output_shape[2] * output_shape[3] + w* output_shape[3] + c )
                    #                new_matrix[var][c*new_shape[1] * new_shape[2] + h* new_shape[2] + w] = matrix[var][h*output_shape[2] * output_shape[3] + w* output_shape[3] + c]
                #else:
                #    new_matrix = matrix
                #print("new matrix ", new_matrix, last_layer, new_matrix.shape)
                network.add_linear(matrix)
                network.add_bias(bias)
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.numlayer+= 1
                num_gpu_layers +=2
                last_layer = "FC"
            elif self.operations[i] == "Gemm":
                
                matrix, bias, m_input_names, b_output_name, b_output_shape = self.resources[i][domain]
                #print("type ", type(matrix), type(bias), matrix.dtype, bias.dtype)
                network.add_linear(matrix.astype("float64"))
                network.add_bias(bias.astype("float64"))
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('FC')
                nn.numlayer+= 1
                #matrix = np.ascontiguousarray(matrix, dtype=np.double)
                #bias = np.ascontiguousarray(bias, dtype=np.double)
                #print("Gemm Matrix ", matrix)
                
                #print("Gemm bias ", bias)
                num_gpu_layers +=2
                last_layer = "FC"
                i += 1
            
            elif self.operations[i] == "Conv2D":
                last_layer = "Conv"
                if i < nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, _, _ = self.resources[i][domain]
                    bias, _, b_output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, b_output_name, b_output_shape = self.resources[i][domain]
                    bias_length = reduce((lambda x, y: x*y), output_shape)
                    bias = np.zeros(bias_length)
                    i += 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[2],image_shape[0],image_shape[1]])
                nn.strides.append([strides[0],strides[1]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.out_shapes.append([b_output_shape[0], b_output_shape[3], b_output_shape[1], b_output_shape[2]])
                nn.filters.append(np.transpose(filters,[3,2,0, 1]))
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                #print("filter shape ", nn.out_shapes[-1])
                network.add_conv_2d(image_shape[0], image_shape[1], filters.astype("float64"), strides[0], [pad_top, pad_left, pad_bottom, pad_right])
                bias=bias.repeat(b_output_shape[1]*b_output_shape[2])
                network.add_bias(bias)
                num_gpu_layers +=2
                nn.numlayer+=1
            elif self.operations[i] == "Conv":
                last_layer = "Conv"
                filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, c_input_names, output_name, b_output_shape = self.resources[i][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[2],image_shape[0],image_shape[1]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append([b_output_shape[0], b_output_shape[3], b_output_shape[1], b_output_shape[2]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.filters.append(np.transpose(filters,[3,2,0, 1]))

                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                nn.numlayer+=1
                #print("Filter Matrix ", filters)
                network.add_conv_2d(image_shape[0], image_shape[1], filters.astype("float64"), strides[0], [pad_top, pad_left, pad_bottom, pad_right])
                bias=bias.repeat(b_output_shape[1]*b_output_shape[2])
                #print("Filter Bias ", bias)
                network.add_bias(bias.astype("float64"))
                num_gpu_layers +=2
                i += 1    
           
            elif self.operations[i] == "Relu":
                #self.resources[i][domain].append(refine)
                nn.layertypes.append('ReLU')
                network.add_relu()
                nn.numlayer += 1
                relu_layers.append(num_gpu_layers)
                num_gpu_layers +=1
                i += 1
            else:
                assert 0, "the optimizer for" + "gpupoly" + " doesn't know of the operation type " + self.operations[i]
        return network, relu_layers, num_gpu_layers
        

    def set_predecessors(self, nn, output):
        output_index_store = {}
        index_o = 0
        for node in output:
            output_index_store[node.output_name] = index_o
            index_o += 1
        for node in output:
            #print("output ", node, node.input_names)
            predecessors = (c_size_t * len(node.input_names))()
            i = 0
            for input_name in node.input_names:
                predecessors[i] = output_index_store[input_name]
                i += 1
            
            node.predecessors = predecessors
            #if not isinstance(node, DeepzonoRelu):
            nn.predecessors.append(predecessors)

    def get_gather_indexes(self, input_shape, indexes, axis):
        size = np.prod(input_shape)
        base_indexes = np.arange(size).reshape(input_shape)
        return np.take(base_indexes, indexes, axis=axis)
