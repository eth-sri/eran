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



class Analyzer:
    def __init__(self, ir_list, domain):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono' or 'deeppoly'
        """
        self.ir_list = ir_list
        self.is_greater = is_greater_zono
        if domain == 'deeppoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        elif domain == 'deepzono':
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
        self.domain = domain
    
    
    def __del__(self):
        elina_manager_free(self.man)
        
    
    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        for i in range(1, len(self.ir_list)):
            element = self.ir_list[i].transformer(self.man, element)
        return element
    
    
    def analyze(self):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element     = self.get_abstract0()
        output_size = 0
        if self.domain == 'deepzono':
            output_size = self.ir_list[-1].output_length
        else:
            output_size = reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)

        #bounds = elina_abstract0_to_box(self.man,element)
        #for i in range(output_size):
        #    print("inf", bounds[i].contents.inf.contents.val.dbl, "sup", bounds[i].contents.sup.contents.val.dbl)
        #elina_interval_array_free(bounds,output_size)
    
        dominant_class = -1
        for i in range(output_size):
            flag = True
            for j in range(output_size):
                if i!=j and not self.is_greater(self.man, element, i, j):
                    flag = False
            if flag:
                dominant_class = i
                break
        
        elina_abstract0_free(self.man, element)
        return dominant_class
    
    
    
    
