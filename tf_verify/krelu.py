import deepzono_nodes as dn
from elina_scalar import *
from elina_dimension import *
from elina_linexpr0 import *
from elina_abstract0 import *
from fppoly import *

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
    def __init__(self, varsid):
        start = time.time()

        # krelu on variables in varsid
        self.varsid = varsid
        self.k = len(varsid)

        cdd_hrepr = self.get_ineqs(varsid)
        check_pt1 = time.time()

        cdd_hrepr = cdd.Matrix(cdd_hrepr, number_type='fraction')
        cdd_hrepr.rep_type = cdd.RepType.INEQUALITY

        pts = self.get_orthant_points(cdd_hrepr)

        # Generate extremal points in the space of variables before and
        # after relu
        pts = [([1] + row + [x if x>0 else 0 for x in row]) for row in pts]

        cdd_vrepr = cdd.Matrix(pts, number_type='fraction')
        cdd_vrepr.rep_type = cdd.RepType.GENERATOR

        # Convert back to H-repr.
        cons = cdd.Polyhedron(cdd_vrepr).get_inequalities()
        cons = np.asarray(cons, dtype=np.float64)

        # normalize constraints for numerical stability
        # more info: http://files.gurobi.com/Numerics.pdf
        absmax = np.absolute(cons).max(axis=1)
        self.cons = cons/absmax[:, None]

        end = time.time()

        return

    def get_ineqs(self, varsid):
        cdd_hrepr = []

        # Get bounds on linear expressions over variables before relu
        # Order of coefficients determined by logic here
        for coeffs in itertools.product([-1, 0, 1], repeat=self.k):
            if all(c==0 for c in coeffs):
                continue

            linexpr0 = generate_linexpr0(self.offset, varsid, coeffs)
            if(self.domain=='refinezono'):
                element = elina_abstract0_assign_linexpr_array(self.man,True,self.element,self.tdim,linexpr0,1,None)
                bound_linexpr = elina_abstract0_bound_dimension(self.man,self.element,self.offset+self.length)
            else:
                bound_linexpr = get_bounds_for_linexpr0(self.man,self.element,linexpr0,self.layerno)
            upper_bound = bound_linexpr.contents.sup.contents.val.dbl

            cdd_hrepr.append([upper_bound] + [-c for c in coeffs])

        return cdd_hrepr

    def get_orthant_points(self, cdd_hrepr):
        # Get points of polytope restricted to all possible orthtants
        pts = []
        for polarity in itertools.product([-1, 1], repeat=self.k):
            hrepr = cdd_hrepr.copy()

            # add constraints to restrict to +ve/-ve half of variables
            for i in range(self.k):
                row = [0]*(self.k+1)
                row[1+i] = polarity[i]
                # row corresponds to the half-space x_i>=0 if polarity[i]==+1
                hrepr.extend([row])

            # remove reduntant constraints
            hrepr.canonicalize()

            # Convert to V-repr.
            pts_new = cdd.Polyhedron(hrepr).get_generators()
            assert all(row[0]==1 for row in pts_new)

            for row in pts_new:
                pts.append(list(row[1:]))

        return pts

def make_krelu_obj(varsid):
    return Krelu(varsid)

# HEURISTIC
def grouping_heuristic(ind, lb, ub):
    # groups elements based on distance in 2D plane
    # each element represented by point (lb[i], ub[i])

    sortlist = sorted(ind, key=lambda k: lb[k]*ub[k])
    return sortlist

    # ret_order = []

    # while ind:
    #     # select element with most L1 norm
    #     x = max(range(len(ind)), key=lambda i: np.abs(ub[ind[i]]*lb[ind[i]]))
    #     indx = ind.pop(x)
    #     ret_order.append(indx)

    #     # select next (k-1) elements closest to x (L2 distance wise)
    #     elems_toadd = 0
    #     if len(ind)>=2 and config.use_3relu:
    #         elems_toadd = 2
    #     elif len(ind)>=1 and config.use_2relu:
    #         elems_toadd = 1

    #     for _ in range(elems_toadd):
    #         y = min(range(len(ind)), key=lambda i: (ub[ind[i]]-ub[indx])**2 + (lb[ind[i]]-lb[indx])**2)
    #         indy = ind.pop(y)
    #         ret_order.append(indy)

    # return ret_order

def encode_krelu_cons(nn, man, element, offset, layerno, length, lbi, ubi, relu_groups, need_pop, domain):

    if(need_pop):
        relu_groups.pop()

    lbi = np.asarray(lbi, dtype=np.double)
    ubi = np.asarray(ubi, dtype=np.double)

    candidate_vars = [i for i in range(length) if lbi[i]<0 and ubi[i]>0]
    candidate_vars = sorted(candidate_vars, key=lambda i: ubi[i]-lbi[i], reverse=True)
    candidate_vars = grouping_heuristic(candidate_vars, lbi, ubi)

    relucons = []

    tdim = ElinaDim(offset+length)
    if domain == 'refinezono':
        element = dn.add_dimensions(man,element,offset+length,1)

    krelu_args = []

    if config.dyn_krelu and candidate_vars:
        limit3relucalls = 500
        firstk = math.sqrt(6*limit3relucalls/len(candidate_vars))
        firstk = int(min(firstk, len(candidate_vars)))
        if(layerno<3):
            firstk = 1
        else:
            firstk = 5#int(max(1,firstk))
        print("firstk ",firstk)
        if firstk>3:
            while candidate_vars:
                headlen = min(firstk, len(candidate_vars))
                head = candidate_vars[:headlen]
                candidate_vars = candidate_vars[headlen:]
                if len(head)<=3:
                    krelu_args.append(head)
                else:
                    for arg in itertools.combinations(head, 3):
                        krelu_args.append(arg)

    klist = ([3] if (config.use_3relu) else []) + ([2] if (config.use_2relu) else []) + [1]

    for k in klist:
        while len(candidate_vars) >= k:
            krelu_args.append(candidate_vars[:k])
            candidate_vars = candidate_vars[k:]
    Krelu.man = man
    Krelu.element = element
    Krelu.tdim = tdim
    Krelu.length = length
    Krelu.layerno = layerno
    Krelu.offset = offset
    Krelu.domain = domain

    start = time.time()
    with multiprocessing.Pool(config.numproc_krelu) as pool:
        krelu_results = pool.map(make_krelu_obj, krelu_args)
    for krelu_inst in krelu_results:
        relucons.append(krelu_inst)
    end = time.time()

    if domain == 'refinezono':
        element = dn.remove_dimensions(man,element,offset+length,1)

    relu_groups.append(relucons)

    return
