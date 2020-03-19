'''
@author: Adrian Hoffmann
'''

from elina_abstract0 import *
from elina_manager import *
from deeppoly_nodes import *
from deepzono_nodes import *
from functools import reduce
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
        self.maxpool_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.maxpool_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, testing = False):
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
            if self.domain == 'deepzono' or self.domain == 'refinezono':
                element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.relu_groups, self.domain=='refinezono', self.timeout_lp, self.timeout_milp, self.use_default_heuristic, self.testing)
            else:
                element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.relu_groups, self.domain=='refinepoly', self.timeout_lp, self.timeout_milp, self.use_default_heuristic, self.testing)

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
            output_size = reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
    
        dominant_class = -1
        if(self.domain=='refinepoly'):

            relu_needed = [1] * self.nn.numlayer
            self.nn.ffn_counter = 0
            self.nn.conv_counter = 0
            self.nn.maxpool_counter = 0
            self.nn.residual_counter = 0
            counter, var_list, model = create_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub,self.relu_groups, self.nn.numlayer, False,relu_needed)
            #model.setParam('Timeout',1000)
            num_var = len(var_list)
            output_size = num_var - counter


        if self.output_constraints is None:
            for i in range(output_size):
                flag = True
                label = i
                for j in range(output_size):
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
                                model.optimize()
                                if model.Status!=2:
                                    model.write("final.mps")
                                    print (f"Model failed to solve, {model.Status}")
                                    flag = False
                                    break
                                elif(model.objval<0):
                                    flag = False
                                    break

                            else:
                                flag = False
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
                    if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1]):
                        or_result = True
                        break

                if not or_result:
                    dominant_class = False
                    break

        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub
