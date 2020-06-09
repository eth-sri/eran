import numpy as np
from zonoml import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from functools import reduce
from ai_milp import *
from krelu import encode_krelu_cons
from config import *
if config.device == Device.CPU:
    from fppoly import *
else:
    from fppoly_gpu import *



def update_relu_expr_bounds(man, element, layerno, bound_expr_list):
    for i in range(len(bound_expr_list)):
        Krelu_expr = bound_expr_list[i]
        varsid = Krelu_expr.varsid
        varsid = np.ascontiguousarray(varsid, dtype=np.uintp)
        lower_bound_expr = Krelu_expr.lexpr
        upper_bound_expr = Krelu_expr.uexpr
        k = len(varsid)
        for j in range(k):
            uexpr = upper_bound_expr[j]
            lexpr = lower_bound_expr[j]
            nnz_l = 0
            nnz_u = 0
            for l in range(k):
                if lexpr[l+1] != 0:
                    nnz_l+=1
                if uexpr[l+1] != 0:
                    nnz_u+=1
            #if nnz_l > 1:
                #lexpr = np.ascontiguousarray(lexpr, dtype=np.double)
                #update_relu_lower_bound_for_neuron(man, element, layerno, varsid[j], lexpr, varsid, k)
            if nnz_u > 1:
                uexpr = np.ascontiguousarray(uexpr, dtype=np.double)
                #update_relu_upper_bound_for_neuron(man, element, layerno, varsid[j], uexpr, varsid, k)

def refine_relu_with_solver_bounds(nn, self, man, element, nlb, nub, relu_groups, timeout_lp, timeout_milp, use_default_heuristic, domain):
    """
    refines the relu transformer

    Arguments
    ---------
    self : Object
        will be a DeepzonoNode or DeeppolyNode
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
    layerno = nn.calc_layerno()
    predecessor_index = nn.predecessors[layerno+1][0] - 1
    if domain == 'deepzono':
        offset, length = self.abstract_information
    else:
        offset = 0
        length = get_num_neurons_in_layer(man, element, predecessor_index)
    lbi = nlb[predecessor_index]
    ubi = nub[predecessor_index]
    if nn.activation_counter==0:
        if domain=='deepzono':
            encode_krelu_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi, relu_groups, False, 'refinezono')
            element = relu_zono_layerwise(man,True,element,offset, length, use_default_heuristic)
            return element
        else:
            bound_expr_list = encode_krelu_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi, relu_groups, False, 'refinepoly')
            handle_relu_layer(*self.get_arguments(man, element), use_default_heuristic)
            update_relu_expr_bounds(man, element, layerno, bound_expr_list)
                   
    else:
        is_conv = False
        timeout = timeout_milp
        for i in range(nn.numlayer):
            if nn.layertypes[i] == 'Conv2D':
                is_conv = True
                break

        if is_conv==True:
            use_milp = 1
        else:
            if predecessor_index<=3:
               use_milp = 1
            else:
               use_milp = 0
               timeout = timeout_lp
        use_milp = use_milp and config.use_milp
        candidate_vars = []
        for i in range(length):
            if((lbi[i]<0 and ubi[i]>0) or (lbi[i]>0)):
                 candidate_vars.append(i)
        #TODO handle residual layers here

        resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, predecessor_index, predecessor_index, length, nlb, nub, relu_groups, use_milp,  candidate_vars, timeout)
        nlb[predecessor_index] = resl
        nub[predecessor_index] = resu

        lbi = nlb[predecessor_index]
        ubi = nub[predecessor_index]
        if domain == 'deepzono':
            encode_krelu_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi, relu_groups, False, 'refinezono')
            

        j = 0
        if domain == 'deepzono':
            for i in range(length):
                if((j < len(indices)) and (i==indices[j])):
             
                    element = relu_zono_refined(man,True, element,i+offset, resl[i],resu[i])
                    j=j+1
                else:
                    element = relu_zono(man,True,element,i+offset)
            return element
        else:
            for j in indices:
                update_bounds_for_neuron(man,element,predecessor_index,j,resl[j],resu[j])
            bound_expr_list = encode_krelu_cons(nn, man, element, offset, predecessor_index, length, lbi, ubi, relu_groups, False, 'refinepoly')
            handle_relu_layer(*self.get_arguments(man, element), use_default_heuristic)
            update_relu_expr_bounds(man, element, layerno, bound_expr_list)
      
     
    
