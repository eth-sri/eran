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


from elina_scalar import *
from elina_dimension import *
from elina_linexpr0 import *
from elina_abstract0 import *
from fppoly import *
from fconv import *

import numpy as np
import time
import itertools
import multiprocessing
import math

from config import config

"""
For representing the constraints CDD format is used
http://web.mit.edu/sage/export/cddlib-094b.dfsg/doc/cddlibman.ps:
each row represents b + Ax >= 0
example: 2*x_1 - 3*x_2 >= 1 translates to [-1, 2, -3]
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


class KAct:
    def __init__(self, input_hrep, approx=True):
        assert KAct.type in ["ReLU", "Tanh", "Sigmoid"]
        self.k = len(input_hrep[0]) - 1
        self.input_hrep = np.array(input_hrep)

        if KAct.type == "ReLU":
            if approx:
                self.cons = fkrelu(self.input_hrep)
            else:
                self.cons = krelu_with_cdd(self.input_hrep)
        elif not approx:
            assert False, "not implemented"
        elif KAct.type == "Tanh":
            self.cons = ftanh_orthant(self.input_hrep)
        else:
            self.cons = fsigm_orthant(self.input_hrep)


def make_kactivation_obj(input_hrep, approx=True):
    return KAct(input_hrep, approx)


def get_ineqs_zono(varsid):
    input_hrep = []

    # Get bounds on linear expressions over variables before relu
    # Order of coefficients determined by logic here
    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
        if all(c==0 for c in coeffs):
            continue

        linexpr0 = generate_linexpr0(KAct.offset, varsid, coeffs)
        element = elina_abstract0_assign_linexpr_array(KAct.man, True, KAct.element,
                                                       KAct.tdim, linexpr0, 1, None)
        bound_linexpr = elina_abstract0_bound_dimension(KAct.man, KAct.element,
                                                        KAct.offset + KAct.length)
        upper_bound = bound_linexpr.contents.sup.contents.val.dbl
        input_hrep.append([upper_bound] + [-c for c in coeffs])
    return input_hrep


def sparse_heuristic_with_cutoff(length, lb, ub, K=3, s=-2):
    assert length == len(lb) == len(ub)

    all_vars = [i for i in range(length) if lb[i] < 0 < ub[i]]
    areas = {var: -lb[var] * ub[var] for var in all_vars}

    assert len(all_vars) == len(areas)
    sparse_n = config.sparse_n
    cutoff = 0.05
    # Sort vars by descending area
    all_vars = sorted(all_vars, key=lambda var: -areas[var])

    vars_above_cutoff = [i for i in all_vars if areas[i] >= cutoff]
    n_vars_above_cutoff = len(vars_above_cutoff)

    kact_args = []
    while len(vars_above_cutoff) > 0 and config.sparse_n >= K:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen <= K:
            kact_args.append(group)
        elif K>2:
            sparsed_combs = generate_sparse_cover(grouplen, K, s=s)
            for comb in sparsed_combs:
                kact_args.append(tuple([group[i] for i in comb]))
        elif K==2:
            raise RuntimeError("K=2 is not supported")

    # Also just apply 1-relu for every var.
    for var in all_vars:
        kact_args.append([var])

    print("krelu: n", config.sparse_n,
          "split_zero", len(all_vars),
          "after cutoff", n_vars_above_cutoff,
          "number of args", len(kact_args))

    return kact_args


def sparse_heuristic_curve(length, lb, ub, is_sigm, s=-2):
    assert length == len(lb) == len(ub)
    all_vars = [i for i in range(length)]
    K = 3
    sparse_n = config.sparse_n
    # Sort vars by descending area

    vars_above_cutoff = all_vars[:]
    vars_above_cutoff = [i for i in vars_above_cutoff if ub[i] - lb[i] >= 0.1]
    limit = 4 if is_sigm else 3
    vars_above_cutoff = [i for i in vars_above_cutoff if lb[i] <= limit and ub[i] >= -limit]
    n_vars_after_cutoff = len(vars_above_cutoff)

    kact_args = []
    while len(vars_above_cutoff) > 0 and config.sparse_n >= K:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen <= K:
            kact_args.append(group)
        else:
            sparsed_combs = generate_sparse_cover(grouplen, K, s=s)
            for comb in sparsed_combs:
                kact_args.append(tuple([group[i] for i in comb]))

    # # Also just apply 1-relu for every var.
    for var in all_vars:
        kact_args.append([var])

    # kact_args = [arg for arg in kact_args if len(arg) == 3]

    print("krelu: n", config.sparse_n,
          "after cutoff", n_vars_after_cutoff,
          "number of args", len(kact_args),
          "Sigm" if is_sigm else "Tanh")

    return kact_args


def encode_kactivation_cons(nn, man, element, offset, layerno, length, lbi, ubi, constraint_groups, need_pop, domain, activation_type, K=3, s=-2, approx=True):
    import deepzono_nodes as dn
    if need_pop:
        constraint_groups.pop()

    lbi = np.asarray(lbi, dtype=np.double)
    ubi = np.asarray(ubi, dtype=np.double)

    if activation_type == "ReLU":
        kact_args = sparse_heuristic_with_cutoff(length, lbi, ubi, K=K, s=s)
    else:
        kact_args = sparse_heuristic_curve(length, lbi, ubi, activation_type == "Sigmoid", s=s)

    kact_cons = []
    tdim = ElinaDim(offset+length)
    if domain == 'refinezono':
        element = dn.add_dimensions(man,element,offset+length,1)

    KAct.man = man
    KAct.element = element
    KAct.tdim = tdim
    KAct.length = length
    KAct.layerno = layerno
    KAct.offset = offset
    KAct.domain = domain
    KAct.type = activation_type

    start = time.time()

    if domain == 'refinezono':
        with multiprocessing.Pool(config.numproc) as pool:
            input_hrep_array = pool.map(get_ineqs_zono, kact_args)
    else:
        total_size = 0
        for varsid in kact_args:
            size = 3**len(varsid) - 1
            total_size = total_size + size

        linexpr0 = elina_linexpr0_array_alloc(total_size)
        i = 0
        for varsid in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue

                linexpr0[i] = generate_linexpr0(offset, varsid, coeffs)
                i = i + 1
        upper_bound = get_upper_bound_for_linexpr0(man,element,linexpr0, total_size, layerno)
        i=0
        input_hrep_array = []
        for varsid in kact_args:
            input_hrep = []
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                input_hrep.append([upper_bound[i]] + [-c for c in coeffs])
                i = i + 1
            input_hrep_array.append(input_hrep)
    end_input = time.time()

    # kact_results = list(map(make_kactivation_obj, input_hrep_array))

    # end1 = time.time()

    with multiprocessing.Pool(config.numproc) as pool:
        # kact_results = pool.map(make_kactivation_obj, input_hrep_array)
        kact_results = pool.starmap(make_kactivation_obj, zip(input_hrep_array, len(input_hrep_array) * [approx]))

    # end2 = time.time()

    # if end2-end_input>10:
    #     print(f"list(map()) time: {end1-end_input:.3f}, multiprocessing time: {end2-end1:.3f}")

    gid = 0
    for inst in kact_results:
        varsid = kact_args[gid]
        inst.varsid = varsid
        kact_cons.append(inst)
        gid = gid+1
    end = time.time()

    if config.debug:
        print(f'total k-activation time: {end-start:.3f}. Time for input: {end_input-start:.3f}. Time for k-activation constraints {end-end_input:.3f}.')
    if domain == 'refinezono':
        element = dn.remove_dimensions(man, element, offset+length, 1)

    constraint_groups.append(kact_cons)
