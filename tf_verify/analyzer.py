'''
@author: Adrian Hoffmann
'''

import numpy as np
from elina_abstract0 import *
from elina_manager import *
from deeppoly_nodes import *
from deepzono_nodes import *
from functools import reduce
import ctypes

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
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.maxpool_counter = 0
        self.maxpool_lb = []
        self.maxpool_ub = []
        self.specLB = []
        self.specUB = []
        self.lastlayer = None

class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, specnumber, use_area_heuristic):
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
        if domain == 'deeppoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        elif domain == 'deepzono' or domain == 'refinezono':
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
        if domain == 'refinezono':
            self.refine = True
        self.domain = domain
        self.nn = nn
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.specnumber = specnumber
        self.use_area_heuristic = use_area_heuristic    

    
    def __del__(self):
        elina_manager_free(self.man)
        
    
    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        for i in range(1, len(self.ir_list)):
            if self.domain == 'deepzono' or self.domain == 'refinezono':
                element = self.ir_list[i].transformer(self.nn, self.man, element, nlb,nub, self.domain=='refinezono', self.timeout_lp, self.timeout_milp)
            else:
                element = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.use_area_heuristic)
        return element, nlb, nub
    
    
    def analyze(self):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element, nlb, nub  = self.get_abstract0()
        output_size = 0
        if self.domain == 'deepzono' or self.domain == 'refinezono':
            output_size = self.ir_list[-1].output_length
        else:
            output_size = reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)

        #bounds = elina_abstract0_to_box(self.man,element)
        #for i in range(output_size):
        #    print("inf", bounds[i].contents.inf.contents.val.dbl, "sup", bounds[i].contents.sup.contents.val.dbl)
        #elina_interval_array_free(bounds,output_size)
    
        dominant_class = -1
        if self.specnumber==0:
            for i in range(output_size):
                flag = True
                for j in range(output_size):
                    if self.domain == 'deepzono' or self.domain == 'refinezono':
                        if i!=j and not self.is_greater(self.man, element, i, j):
                            flag = False
                    else:
                        if i!=j and not self.is_greater(self.man, element, i, j, self.use_area_heuristic):
                            flag = False
                if flag:
                    dominant_class = i
                    break
        elif self.specnumber==9:
            flag = True
            for i in range(output_size):
                if self.domain == 'deepzono' or self.domain == 'refinezono':
                    if i!=3 and not self.is_greater(self.man, element, i, 3):
                        flag = False
                        break
                else:
                    if i!=3 and not self.is_greater(self.man, element, i, 3, self.use_area_heuristic):
                        flag = False
                        break
            if flag:
                dominant_class = 3
                
        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub
    
    
    
    
