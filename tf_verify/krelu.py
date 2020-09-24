from elina_scalar import *
from elina_dimension import *
from elina_linexpr0 import *
from elina_abstract0 import *
from fppoly import *
from fconv import *

import numpy as np
import cdd
import time
import itertools
import multiprocessing
import math

from config import config

"""
From reference manual:
http://web.mit.edu/sage/export/cddlib-094b.dfsg/doc/cddlibman.ps

CDD H-representaion format:
each row represents b + Ax >= 0
example: 2*x_1 - 3*x_2 >= 1 translates to [-1, 2, -3]

CDD V-representaion format:
<code> x_1 x_2 ... x_d
code is 1 for extremal point, 0 for generating direction (rays)
example: extreme point (2, -1, 3) translates to [1, 2, -1, 3]
all polytopes generated here should be closed, hence code=1
"""

def generate_linexpr0(offset, varids, coeffs):
    # returns ELINA expression, equivalent to sum_i(varids[i]*coeffs[i])
    assert len(varids) == len(coeffs)
    n = len(varids)

    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, n)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, 0)

    for i, (x, coeffx) in enumerate(zip(varids, coeffs)):
        linterm = pointer(linexpr0.contents.p.linterm[i])
        linterm.contents.dim = ElinaDim(offset + x)
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_double(coeff.contents.val.scalar, coeffx)

    return linexpr0

class Krelu:
    def __init__(self, cdd_hrepr):
        start = time.time()

        # krelu on variables in varsid
        #self.varsid = varsid
        self.k = len(cdd_hrepr[0])-1
        self.cdd_hrepr = np.array(cdd_hrepr)
        self.cons = fkrelu(self.cdd_hrepr)

        return


def make_krelu_obj(varsid):
    return Krelu(varsid)


class Krelu_expr:
    def __init__(self, expr, varsid, bound):
        self.expr = expr
        self.varsid = varsid
        self.bound = bound


def get_ineqs_zono(varsid):
    cdd_hrepr = []

    # Get bounds on linear expressions over variables before relu
    # Order of coefficients determined by logic here
    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
        if all(c==0 for c in coeffs):
            continue

        linexpr0 = generate_linexpr0(Krelu.offset, varsid, coeffs)
        element = elina_abstract0_assign_linexpr_array(Krelu.man,True,Krelu.element,Krelu.tdim,linexpr0,1,None)
        bound_linexpr = elina_abstract0_bound_dimension(Krelu.man,Krelu.element,Krelu.offset+Krelu.length)
        upper_bound = bound_linexpr.contents.sup.contents.val.dbl
        cdd_hrepr.append([upper_bound] + [-c for c in coeffs])
    return cdd_hrepr


def compute_bound(constraint, lbi, ubi, varsid, j, is_lower):
    k = len(varsid)
    divisor = -constraint[j+k+1]
    actual_bound = constraint[0]/divisor
    potential_improvement = 0
    for l in range(k):
        coeff = constraint[l+1]/divisor
        if is_lower:
           if coeff < 0:
               actual_bound += coeff * ubi[varsid[l]]
           elif coeff > 0:
               actual_bound += coeff * lbi[varsid[l]]
        else:
           if coeff < 0:
               actual_bound += coeff * lbi[varsid[l]]
           elif coeff > 0:
               actual_bound += coeff * ubi[varsid[l]]
        potential_improvement += abs(coeff * (ubi[varsid[l]] - lbi[varsid[l]]))
        if l==j:
            continue
        coeff = constraint[l+k+1]/divisor
        if((is_lower and coeff<0) or ((not is_lower) and (coeff > 0))):
            actual_bound += coeff * ubi[varsid[l]]
    return actual_bound, potential_improvement

def calculate_nnz(constraint, k):
    nnz = 0
    for i in range(k):
        if constraint[i+1] != 0:
            nnz = nnz+1
    return nnz            

def compute_expr_bounds_from_candidates(krelu_inst, varsid, bound_expr, lbi, ubi, candidate_bounds, is_lower):
    assert not is_lower
    k = krelu_inst.k
    cons = krelu_inst.cons
    for j in range(k):
        candidate_rows = candidate_bounds[j]
        if is_lower:
            best_bound = -math.inf
        else:
            best_bound = math.inf
        best_index = -1
        for i in range(len(candidate_rows)):
            row_index = candidate_rows[i]
            actual_bound, potential_improvement = compute_bound(cons[row_index], lbi, ubi, varsid, j, is_lower)
            bound = actual_bound - potential_improvement / 2
            nnz = calculate_nnz(cons[row_index], k)
            if nnz < 2:
                continue
            if((is_lower and bound > best_bound) or ((not is_lower) and bound < best_bound)):
                best_index = row_index
                best_bound = bound
        if best_index == -1:
            continue
        res = np.zeros(k+1)
        best_row = cons[best_index]
        divisor = -best_row[j+k+1]
        assert divisor > 0
        #if divisor == 0:
        #    print("ROW ",best_row)
        #    print("CONS ", cons, krelu_inst)
        #    print("CDD ", krelu_inst.cdd_hrepr)
        #    print("j ", j, "lb ", lbi[varsid[0]], lbi[varsid[1]], "ub ", ubi[varsid[0]], ubi[varsid[1]] )
        #    print("candidates ", len(candidate_rows))
        res[0] = best_row[0]/divisor
        for l in range(k):
            res[l+1] = best_row[l+1]/divisor
            if(l==j):
                continue
            coeff = best_row[l+k+1]/divisor
            if((is_lower and coeff<0) or ((not is_lower) and (coeff > 0))):
                res[0] = res[0] + coeff*ubi[varsid[l]]  
                print("res ", res, "best_row ", best_row,"j ", j)
        if varsid[j] in bound_expr.keys():
            current_bound = bound_expr[varsid[j]].bound
            if (is_lower and best_bound > current_bound) or ((not is_lower) and best_bound < current_bound):
                    bound_expr[varsid[j]] = Krelu_expr(res, varsid, best_bound)
        else:
            bound_expr[varsid[j]] = Krelu_expr(res, varsid, best_bound)

def compute_expr_bounds(krelu_inst, varsid, lower_bound_expr, upper_bound_expr, lbi, ubi):
    cons = krelu_inst.cons
    nbrows = len(cons)
    k = len(varsid)
    candidate_lower_bounds = []
    candidate_upper_bounds = []
    for j in range(k):
        candidate_lower_bounds.append([])
        candidate_upper_bounds.append([])
    lin_size = len(krelu_inst.lin_set)
    new_cons = np.zeros((lin_size,2*k+1),dtype=np.float64)
    lin_count = 0
    for i in range(nbrows):
        if i in krelu_inst.lin_set:
            row = cons[i]
            for j in range(2*k+1):
                new_cons[lin_count][j] = -row[j]
            lin_count = lin_count + 1

    krelu_inst.cons = np.vstack([cons,new_cons])
    cons = krelu_inst.cons
    nbrows = len(cons)
    for i in range(nbrows):
        row = cons[i]
        for j in range(k):
            if row[j+k+1]<0:
                candidate_upper_bounds[j].append(i)
            elif row[j+k+1]>0:
                candidate_lower_bounds[j].append(i)
    #compute_expr_bounds_from_candidates(krelu_inst, varsid, lower_bound_expr, lbi, ubi, candidate_lower_bounds, True)
    compute_expr_bounds_from_candidates(krelu_inst, varsid, upper_bound_expr, lbi, ubi, candidate_upper_bounds, False)


def sparse_heuristic_with_cutoff(all_vars, areas):
    assert len(all_vars) == len(areas)
    K = 3
    sparse_n = config.sparse_n
    cutoff = 0.05
    print("sparse n", sparse_n)
    # Sort vars by descending area
    all_vars = sorted(all_vars, key=lambda var: -areas[var])

    vars_above_cutoff = [i for i in all_vars if areas[i] >= cutoff]

    krelu_args = []
    while len(vars_above_cutoff) > 0:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen <= K:
            krelu_args.append(group)
        else:
            sparsed_combs = generate_sparse_cover(grouplen, K)
            for comb in sparsed_combs:
                krelu_args.append(tuple([group[i] for i in comb]))

    # Also just apply 1-relu for every var.
    for var in all_vars:
        krelu_args.append([var])

    return krelu_args


def encode_kactivation_cons(nn, man, element, offset, layerno, length, lbi, ubi, relu_groups, need_pop, domain, activation_type):
    import deepzono_nodes as dn
    if(need_pop):
        relu_groups.pop()

    last_conv = -1
    is_conv = False
    for i in range(nn.numlayer):
        if nn.layertypes[i] == 'Conv':
            last_conv = i
            is_conv = True

    lbi = np.asarray(lbi, dtype=np.double)
    ubi = np.asarray(ubi, dtype=np.double)

    candidate_vars = [i for i in range(length) if lbi[i] < 0 and ubi[i] > 0]
    candidate_vars_areas = {var: -lbi[var] * ubi[var] for var in candidate_vars}
    # Sort vars by descending area
    candidate_vars = sorted(candidate_vars, key=lambda var: -candidate_vars_areas[var])

    # Use sparse heuristic to select args (uncomment to use)
    krelu_args = sparse_heuristic_with_cutoff(candidate_vars, candidate_vars_areas)

    relucons = []
    #print("UBI ",ubi)
    tdim = ElinaDim(offset+length)
    if domain == 'refinezono':
        element = dn.add_dimensions(man,element,offset+length,1)

    #krelu_args = []
    #if config.dyn_krelu and candidate_vars:
    #    limit3relucalls = 500
    #    firstk = math.sqrt(6*limit3relucalls/len(candidate_vars))
    #    firstk = int(min(firstk, len(candidate_vars)))
    #    if is_conv and layerno < last_conv:
    #        firstk = 1
    #    else:
    #        firstk = 5#int(max(1,firstk))
    #    print("firstk ",firstk)
    #    if firstk>3:
    #        while candidate_vars:
    #            headlen = min(firstk, len(candidate_vars))
    #            head = candidate_vars[:headlen]
    #            candidate_vars = candidate_vars[headlen:]
    #            if len(head)<=3:
    #               krelu_args.append(head)
    #            else:
    #                for arg in itertools.combinations(head, 3):
    #                    krelu_args.append(arg)

    #klist = ([3] if (config.use_3relu) else []) + ([2] if (config.use_2relu) else []) + [1]
    #for k in klist:
    #    while len(candidate_vars) >= k:
    #        krelu_args.append(candidate_vars[:k])
    #        candidate_vars = candidate_vars[k:]
    Krelu.man = man
    Krelu.element = element
    Krelu.tdim = tdim
    Krelu.length = length
    Krelu.layerno = layerno
    Krelu.offset = offset
    Krelu.domain = domain

    start = time.time()
    if domain == 'refinezono':
        with multiprocessing.Pool(config.numproc) as pool:
            cdd_hrepr_array = pool.map(get_ineqs_zono, krelu_args)    
    else:
    #    krelu_results = []
        total_size = 0
        for varsid in krelu_args:
            size = 3**len(varsid) - 1
            total_size = total_size + size

        linexpr0 = elina_linexpr0_array_alloc(total_size)
        i = 0
        for varsid in krelu_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c==0 for c in coeffs):
                    continue

                linexpr0[i] = generate_linexpr0(offset, varsid, coeffs)
                i = i + 1
        upper_bound = get_upper_bound_for_linexpr0(man,element,linexpr0, total_size, layerno)
        i=0
        cdd_hrepr_array = []
        for varsid in krelu_args:
            cdd_hrepr = []
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c==0 for c in coeffs):
                    continue
                cdd_hrepr.append([upper_bound[i]] + [-c for c in coeffs])
                #print("UPPER BOUND ", upper_bound[i], "COEFF ", coeffs)
                #if len(varsid)>1:
                #    print("LB ", lbi[varsid[0]],lbi[varsid[1]], "UB ", ubi[varsid[0]], ubi[varsid[1]])
                i = i + 1
            cdd_hrepr_array.append(cdd_hrepr)

    with multiprocessing.Pool(config.numproc) as pool:
        krelu_results = pool.map(make_krelu_obj, cdd_hrepr_array)


    #        krelu_results.append(make_krelu_obj(krelu_args[i]))
    #bound_expr_list = [] 
    gid = 0
    lower_bound_expr = {}
    upper_bound_expr = {}
    for krelu_inst in krelu_results:
        varsid = krelu_args[gid]
        krelu_inst.varsid = varsid
        # For now disabling since in the experiments updating expression bounds makes results worse.
        # compute_expr_bounds(krelu_inst, varsid, lower_bound_expr, upper_bound_expr, lbi, ubi)
        #print("VARSID ",varsid)
        #bound_expr_list.append(Krelu_expr(lower_bound_expr, upper_bound_expr, varsid))
        relucons.append(krelu_inst)
        gid = gid+1
    end = time.time()

    if config.debug:
        print('krelu time spent: ' + str(end-start))
    if domain == 'refinezono':
        element = dn.remove_dimensions(man,element,offset+length,1)

    relu_groups.append(relucons)

    return lower_bound_expr, upper_bound_expr
