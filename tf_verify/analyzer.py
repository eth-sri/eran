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


'''
@author: Adrian Hoffmann
'''

from elina_abstract0 import *
from elina_manager import *
from deeppoly_nodes import *
from deepzono_nodes import *
from functools import reduce
from ai_milp import milp_callback
import gc

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pool_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l,w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l,w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradiant(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper




class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing = False):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono', 'refinezono' or 'deeppoly'
        """
        self.ir_list = ir_list
        self.is_greater = is_greater_zono
        self.refine = False
        if domain == 'deeppoly' or domain == 'refinepoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        elif domain == 'deepzono' or domain == 'refinezono':
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
        if domain == 'refinezono' or domain == 'refinepoly':
            self.refine = True
        self.domain = domain
        self.nn = nn
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.output_constraints = output_constraints
        self.use_default_heuristic = use_default_heuristic
        self.testing = testing
        self.relu_groups = []
        self.label = label
        self.prop = prop
    
    def __del__(self):
        elina_manager_free(self.man)
        
    
    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        testing_nlb = []
        testing_nub = []
        for i in range(1, len(self.ir_list)):
            element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.relu_groups, 'refine' in self.domain, self.timeout_lp, self.timeout_milp, self.use_default_heuristic, self.testing)

            if self.testing and isinstance(element_test_bounds, tuple):
                element, test_lb, test_ub = element_test_bounds
                testing_nlb.append(test_lb)
                testing_nub.append(test_ub)
            else:
                element = element_test_bounds
        if self.domain in ["refinezono", "refinepoly"]:
            gc.collect()
        if self.testing:
            return element, testing_nlb, testing_nub
        return element, nlb, nub
    
    
    def analyze(self):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element, nlb, nub = self.get_abstract0()
        output_size = 0
        if self.domain == 'deepzono' or self.domain == 'refinezono':
            output_size = self.ir_list[-1].output_length
        else:
            output_size = self.ir_list[-1].output_length#reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
    
        dominant_class = -1
        if(self.domain=='refinepoly'):

            #relu_needed = [1] * self.nn.numlayer
            self.nn.ffn_counter = 0
            self.nn.conv_counter = 0
            self.nn.pool_counter = 0
            self.nn.residual_counter = 0
            self.nn.activation_counter = 0
            counter, var_list, model = create_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub,self.relu_groups, self.nn.numlayer, config.complete==True)
            if config.complete==True:
                model.setParam(GRB.Param.TimeLimit,self.timeout_milp)
            else:
                model.setParam(GRB.Param.TimeLimit,self.timeout_lp)
            num_var = len(var_list)
            output_size = num_var - counter

        label_failed = []
        x = None
        if self.output_constraints is None:
            candidate_labels = []
            if self.label == -1:
                for i in range(output_size):
                    candidate_labels.append(i)
            else:
                candidate_labels.append(self.label)
            adv_labels = []
            if self.prop == -1:
                for i in range(output_size):
                    adv_labels.append(i)
            else:
                adv_labels.append(self.prop)   
            for i in candidate_labels:
                flag = True
                label = i
                for j in adv_labels:
                    if self.domain == 'deepzono' or self.domain == 'refinezono':
                        if i!=j and not self.is_greater(self.man, element, i, j):
                            flag = False
                            break
                    else:
                        if label!=j and not self.is_greater(self.man, element, label, j, self.use_default_heuristic):

                            if(self.domain=='refinepoly'):
                                obj = LinExpr()
                                obj += 1*var_list[counter+label]
                                obj += -1*var_list[counter + j]
                                model.setObjective(obj,GRB.MINIMIZE)
                                if config.complete == True:
                                    model.optimize(milp_callback)
                                    if model.objbound <= 0:
                                        flag = False
                                        if self.label!=-1:
                                            label_failed.append(j)
                                        if model.solcount > 0:
                                            x = model.x[0:len(self.nn.specLB)]
                                        break    
                                else:
                                    model.optimize()
                                    if model.Status!=2:
                                        model.write("final.mps")
                                        flag = False
                                        break
                                    elif model.objval < 0:
                               
                                        flag = False
                                        if model.objval != math.inf:
                                            x = model.x[0:len(self.nn.specLB)]
                                        break

                            else:
                                flag = False
                                if self.label!=-1:
                                    label_failed.append(j)
                                if config.complete == False:
                                    break


                if flag:
                    dominant_class = i
                    break
        else:
            # AND
            dominant_class = True
            for or_list in self.output_constraints:
                # OR
                or_result = False
                
                for is_greater_tuple in or_list:
                    if is_greater_tuple[1] == -1:
                        if nub[-1][is_greater_tuple[0]] <= float(is_greater_tuple[2]):
                            or_result = True
                            break
                    else: 
                        if self.domain == 'deepzono' or self.domain == 'refinezono':
                            if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1]):
                                or_result = True
                                break
                        else:
                            if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1], self.use_default_heuristic):
                                or_result = True
                                break

                if not or_result:
                    dominant_class = False
                    break
        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub, label_failed, x
