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


from gurobipy import *
from fconv import *
import numpy as np
from config import config
import multiprocessing
import math
import sys
import time
import warnings
from utils import evaluate_cstr


def milp_callback(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_bound > 0.01:
            model.terminate()
        if obj_best < -0.01:
            model.terminate()

def milp_callback_adex(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        sol_count = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if obj_bound > 0.01:
            model.terminate()
        if obj_best < -0.01 and sol_count > 0:
            model.terminate()

def lp_callback(model, where):
    # pass
    if where == GRB.Callback.SIMPLEX:
        obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
        if model.cbGet(GRB.Callback.SPX_PRIMINF) == 0 and obj_best < -0.01: # and model.cbGet(GRB.Callback.SPX_DUALINF) == 0:
            print("Used simplex terminate")
            model.terminate()
    if where == GRB.Callback.BARRIER:
        obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
        if model.cbGet(GRB.Callback.BARRIER_PRIMINF) == 0  and obj_best < -0.01: # and model.cbGet(GRB.Callback.BARRIER_DUALINF) == 0
            model.terminate()
            print("Used barrier terminate")


def handle_conv(model, var_list, start_counter, filters,biases,filter_size,input_shape, strides, out_shape, pad_top,
                pad_left, pad_bottom, pad_right, lbi, ubi, is_nchw=False, feas_act=None):

    num_out_neurons = np.prod(out_shape)
    num_in_neurons = np.prod(input_shape)#input_shape[0]*input_shape[1]*input_shape[2]
    #print("filters", filters.shape, filter_size, input_shape, strides, out_shape, pad_top, pad_left)
    start_idx = len(var_list)
    for j in range(num_out_neurons):
        var_name = "x" + str(start_idx + j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub =ubi[j], name=var_name)
        var_list.append(var)

    if feas_act is not None:
        for j, var in enumerate(var_list[-num_out_neurons:]):
            var.start = feas_act[j]

    #print("OUT SHAPE ", out_shape, input_shape, filter_size, filters.shape, biases.shape)
    if is_nchw:
        for out_z in range(out_shape[1]):
            for out_x in range(out_shape[2]):
                for out_y in range(out_shape[3]):
                
                    dst_ind = out_z*out_shape[2]*out_shape[3] + out_x*out_shape[3] + out_y
                    expr = LinExpr()
                    #print("dst ind ", dst_ind)
                    expr += -1*var_list[start_idx + dst_ind]
                    
                    for inp_z in range(input_shape[0]):
                        for x_shift in range(filter_size[0]):
                            for y_shift in range(filter_size[1]):
                                x_val = out_x*strides[0]+x_shift-pad_top
                                y_val = out_y*strides[1]+y_shift-pad_left
                                if(y_val<0 or y_val >= input_shape[2]):
                                    continue
                                if(x_val<0 or x_val >= input_shape[1]):
                                    continue
                                mat_offset = x_val*input_shape[2] + y_val + inp_z*input_shape[1]*input_shape[2]
                                if(mat_offset>=num_in_neurons):
                                    continue 
                                src_ind = start_counter + mat_offset
                                #print("src ind ", mat_offset)
                                #filter_index = x_shift*filter_size[1]*input_shape[0]*out_shape[1] + y_shift*input_shape[0]*out_shape[1] + inp_z*out_shape[1] + out_z
                                expr.addTerms(filters[out_z][inp_z][x_shift][y_shift],var_list[src_ind])
                                                           
                    expr.addConstant(biases[out_z])
                    
                    model.addConstr(expr, GRB.EQUAL, 0)  
                      
    else:
        for out_x in range(out_shape[1]):
            for out_y in range(out_shape[2]):
                for out_z in range(out_shape[3]):
                    dst_ind = out_x*out_shape[2]*out_shape[3] + out_y*out_shape[3] + out_z
                    expr = LinExpr()
                    expr += -1*var_list[start_idx + dst_ind]
                    for inp_z in range(input_shape[2]):
                        for x_shift in range(filter_size[0]):
                            for y_shift in range(filter_size[1]):
                                x_val = out_x*strides[0]+x_shift-pad_top
                                y_val = out_y*strides[1]+y_shift-pad_left

                                if(y_val<0 or y_val >= input_shape[1]):
                                    continue

                                if(x_val<0 or x_val >= input_shape[0]):
                                    continue

                                mat_offset = x_val*input_shape[1]*input_shape[2] + y_val*input_shape[2] + inp_z
                                if(mat_offset>=num_in_neurons):
                                    continue
                                src_ind = start_counter + mat_offset
                                #filter_index = x_shift*filter_size[1]*input_shape[2]*out_shape[3] + y_shift*input_shape[2]*out_shape[3] + inp_z*out_shape[3] + out_z
                             #expr.addTerms(filters[filter_index],var_list[src_ind])
                                expr.addTerms(filters[x_shift][y_shift][inp_z][out_z],var_list[src_ind])

                    expr.addConstant(biases[out_z])
                    model.addConstr(expr, GRB.EQUAL, 0)
    return start_idx


def handle_padding(model, var_list, start_counter, input_shape, out_shape, pad_top, pad_left, pad_bottom, pad_right,
                   lbi, ubi, is_nchw=False, feas_act=None):
    num_out_neurons = np.prod(out_shape)
    num_in_neurons = np.prod(input_shape)  # input_shape[0]*input_shape[1]*input_shape[2]
    # print("filters", filters.shape, filter_size, input_shape, strides, out_shape, pad_top, pad_left)
    start_idx = len(var_list)
    for j in range(num_out_neurons):
        var_name = "x" + str(start_idx + j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j], name=var_name)
        var_list.append(var)

    if feas_act is not None:
        for j, var in enumerate(var_list[-num_out_neurons:]):
            var.start = feas_act[j]

    # print("OUT SHAPE ", out_shape, input_shape, filter_size, filters.shape, biases.shape)
    if is_nchw:
        for out_z in range(out_shape[1]):
            for out_x in range(out_shape[2]):
                for out_y in range(out_shape[3]):

                    dst_ind = out_z * out_shape[2] * out_shape[3] + out_x * out_shape[3] + out_y
                    expr = LinExpr()
                    expr += -1 * var_list[start_idx + dst_ind]

                    x_val = out_x  - pad_top
                    y_val = out_y  - pad_left
                    mat_offset = out_z * input_shape[1] * input_shape[2] + x_val * input_shape[2] + y_val

                    if (y_val < 0 or y_val >= input_shape[2]):
                        pass
                    elif (x_val < 0 or x_val >= input_shape[1]):
                        pass
                    elif (mat_offset >= num_in_neurons):
                        pass
                    else:
                        expr += 1 * var_list[start_counter + mat_offset]
                    model.addConstr(expr, GRB.EQUAL, 0)

    else:
        for out_x in range(out_shape[1]):
            for out_y in range(out_shape[2]):
                for out_z in range(out_shape[3]):
                    dst_ind = out_x * out_shape[2] * out_shape[3] + out_y * out_shape[3] + out_z
                    expr = LinExpr()
                    expr += -1 * var_list[start_idx + dst_ind]
                    x_val = out_x - pad_top
                    y_val = out_y - pad_left
                    mat_offset = x_val * input_shape[1] * input_shape[2] + y_val * input_shape[2] + out_z

                    if (y_val < 0 or y_val >= input_shape[1]):
                        pass
                    elif (x_val < 0 or x_val >= input_shape[0]):
                        pass
                    elif (mat_offset >= num_in_neurons):
                        pass
                    else:
                        expr += 1 * var_list[start_counter + mat_offset]

                    model.addConstr(expr, GRB.EQUAL, 0)
    return start_idx

def handle_maxpool(model, var_list, layerno, src_counter, pool_size, input_shape, strides, output_shape, pad_top,
                   pad_left, lbi, ubi, lbi_prev, ubi_prev, use_milp, is_nchw=False, feas_act=None):
    use_milp = use_milp and config.use_milp

    start_idx = len(var_list)
    num_neurons = np.prod(input_shape)#input_shape[0]*input_shape[1]*input_shape[2]
    binary_counter = start_idx
    maxpool_counter = start_idx

    if(use_milp==1):
        maxpool_counter = start_idx + num_neurons
        for j in range(num_neurons):
            var_name = "x" + str(start_idx + j)
            var = model.addVar(vtype=GRB.BINARY, name=var_name)
            var_list.append(var)

    o1 = output_shape[1]
    o2 = output_shape[2]
    o3 = output_shape[3]
    output_size = o1*o2*o3
    i12 = input_shape[1]*input_shape[2]
    o12 = output_shape[2]*output_shape[3]

    #print("strides ", strides, pad_top, pad_left)
    for j in range(output_size):
        var_name = "x" + str(maxpool_counter+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j],  name=var_name)
        var_list.append(var)

    if feas_act is not None:
        for j, var in enumerate(var_list[-output_size:]):
            var.start = feas_act[j]

    for out_pos in range(output_size):
        if is_nchw:
            out_z = int(out_pos / o12)
            out_x = int((out_pos - out_z * o12) / output_shape[3])
            out_y = int(out_pos - out_z * o12 - out_x * output_shape[3])
        else:
            out_x = int(out_pos / o12)
            out_y = int((out_pos-out_x*o12) / output_shape[3])
            out_z = int(out_pos-out_x*o12 - out_y*output_shape[3])
        inp_z = out_z
               
        max_u = float("-inf")
        max_l = float("-inf")
        sum_l = 0.0
        max_l_var = 0
        max_u_var = 0
        pool_map = []
        l = 0
        for x_shift in range(pool_size[0]):
            for y_shift in range(pool_size[1]):
                x_val = out_x*strides[0] + x_shift - pad_top
                if(x_val<0 or x_val>=input_shape[0]):
                    continue
                y_val = out_y*strides[1] + y_shift - pad_left
                if(y_val < 0 or y_val>=input_shape[1]):
                    continue
                if is_nchw:
                    pool_cur_dim = x_val * input_shape[2] + y_val + inp_z * i12
                else:
                    pool_cur_dim = x_val * i12 + y_val * input_shape[2] + inp_z
                if pool_cur_dim >= num_neurons:
                    continue    
                pool_map.append(pool_cur_dim)
                lb = lbi_prev[pool_cur_dim] 
                ub = ubi_prev[pool_cur_dim]
                sum_l = sum_l + lb       
                if ub > max_u:
                    max_u = ub
                    max_u_var = pool_cur_dim
                if lb > max_l:   
                    max_l = lb
                    max_l_var = pool_cur_dim
                l = l + 1     
        dst_index = maxpool_counter + out_pos
                
        if use_milp==1:
            binary_expr = LinExpr()
            for l in range(len(pool_map)):
                src_index = pool_map[l]
                src_var = src_index + src_counter
                binary_var = src_index + binary_counter
                if(ubi_prev[src_index]<max_l):
                    continue

                # y >= x
                expr = var_list[dst_index] - var_list[src_var]
                model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # y <= x + (1-a)*(u_{rest}-l)
                max_u_rest = float("-inf")
                for j in range(len(pool_map)):
                    if j==l:
                        continue
                    if(ubi_prev[j]>max_u_rest):
                        max_u_rest = ubi_prev[j]

                cst = max_u_rest-lbi_prev[l]
                expr = var_list[dst_index] - var_list[src_var] + cst*var_list[binary_var]
                model.addConstr(expr, GRB.LESS_EQUAL, cst)

	            # indicator constraints
                model.addGenConstrIndicator(var_list[binary_var], True, var_list[dst_index]-var_list[src_var], GRB.EQUAL, 0.0)
                binary_expr += var_list[binary_var]

            # only one indicator can be true
            model.addConstr(binary_expr, GRB.EQUAL, 1)

        else:
            flag = True
            for l in range(len(pool_map)):
                if pool_map[l] == max_l_var:
                    continue
                ub = ubi_prev[pool_map[l]]
                if ub >= max_l:
                   flag = False
                   break
            if flag:
                # one variable dominates all others
                src_var = max_l_var + src_counter
                expr = var_list[dst_index] - var_list[src_var]
                model.addConstr(expr, GRB.EQUAL, 0)
            else:
                # No one variable dominates all other
                add_expr = LinExpr()
                add_expr += -var_list[dst_index]
                for l in range(len(pool_map)):
                    src_index = pool_map[l]
                    src_var = src_index + src_counter
                    # y >= x
                    expr = var_list[dst_index] - var_list[src_var]
                    model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                    add_expr += var_list[src_var]
                model.addConstr(add_expr, GRB.GREATER_EQUAL, sum_l - max_l)

    return maxpool_counter


def handle_affine(model,var_list,counter,weights,biases, lbi, ubi, feas_act=None):
    num_neurons_affine = len(weights)
    start_idx = len(var_list)

    # output of matmult
    for j in range(num_neurons_affine):
        var_name = "x" + str(start_idx + j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j], name=var_name)
        var_list.append(var)

    if feas_act is not None:
        for j, var in enumerate(var_list[-num_neurons_affine:]):
            var.start = feas_act[j]

    for j in range(num_neurons_affine):
        num_in_neurons = len(weights[j])

        expr = LinExpr()
        expr += -1*var_list[start_idx + j]
        # matmult constraints
        for k in range(num_in_neurons):
            expr.addTerms(weights[j][k], var_list[counter+k])
        expr.addConstant(biases[j])
        model.addConstr(expr, GRB.EQUAL, 0)
    return start_idx

def handle_bias(model,var_list,counter,biases, lbi, ubi, feas_act=None):
    num_neurons_affine = len(biases)
    start_idx = len(var_list)

    # output of matmult
    for j in range(num_neurons_affine):
        var_name = "x" + str(start_idx + j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j], name=var_name)
        var_list.append(var)

    if feas_act is not None:
        for j, var in enumerate(var_list[-num_neurons_affine:]):
            var.start = feas_act[j]

    for j in range(num_neurons_affine):
        expr = LinExpr()
        expr += -1*var_list[start_idx + j]
        expr.addTerms(1.0, var_list[counter+j])
        expr.addConstant(biases[j])
        model.addConstr(expr, GRB.EQUAL, 0)
    return start_idx


def handle_residual(model, var_list, branch1_counter, branch2_counter, lbi, ubi, feas_act=None):
    num_neurons_affine = len(lbi)
    start_idx = len(var_list)

    # output of matmult
    for j in range(num_neurons_affine):
        var_name = "x" + str(start_idx + j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub=ubi[j], name=var_name)
        var_list.append(var)

    if feas_act is not None:
        for j, var in enumerate(var_list[-num_neurons_affine:]):
            var.start = feas_act[j]

    for j in range(num_neurons_affine):
        # num_in_neurons = len(weights[j])

        expr = LinExpr()
        expr += -1 * var_list[start_idx + j]
        # matmult constraints
        # for k in range(num_in_neurons):
        expr += var_list[branch1_counter + j]
        expr += var_list[branch2_counter + j]
        expr.addConstant(0)
        model.addConstr(expr, GRB.EQUAL, 0)
    return start_idx


def _add_kactivation_constraints(model, var_list, constraint_groups, x_counter, y_counter):
    #print("start here ")
    for inst in constraint_groups:
        for row in inst.cons:
            k = len(inst.varsid)
            expr = LinExpr()
            expr.addConstant(row[0])
            for i, x in enumerate(inst.varsid):
                expr.addTerms(row[1 + i], var_list[x_counter + x])
                expr.addTerms(row[1 + k + i], var_list[y_counter + x])
            #print("row ", row)
            model.addConstr(expr >= 0)


def handle_relu(model,var_list, affine_counter, num_neurons, lbi, ubi, relu_groupsi, use_milp, partial_milp_neurons, feas_act=None):
    use_milp = use_milp and config.use_milp
    #print("relu groups ")

    start_idx = len(var_list)
    relu_counter = start_idx

    cross_over_idx = list(np.nonzero(np.array(lbi)*np.array(ubi)<0)[0])
    width = np.array(ubi) - np.array(lbi)
    cross_over_idx = sorted(cross_over_idx, key= lambda x: -width[x])

    milp_encode_idx = cross_over_idx[:partial_milp_neurons] if partial_milp_neurons>=0 else cross_over_idx #cross_over_idx if use_milp else cross_over_idx[:partial_milp_neurons]
    temp_idx = np.ones(num_neurons, dtype=bool)
    temp_idx[milp_encode_idx] = False
    relax_encode_idx = np.arange(num_neurons)[temp_idx]

    assert len(relax_encode_idx) + len(milp_encode_idx) == num_neurons

    # #print("neurons ", num_neurons)
    if len(milp_encode_idx)>0:
        init_bin = None if feas_act is None else (feas_act>0).astype(np.float)
        for i, j in enumerate(milp_encode_idx):
            var_name = "x_bin_" + str(start_idx + i)
            var_bin = model.addVar(vtype=GRB.BINARY, name=var_name)
            if init_bin is not None:
                var_bin.start = init_bin[j]
            var_list.append(var_bin)
            relu_counter += 1

    # relu output variables
    for j in range(num_neurons):
        var_name = "x" + str(relu_counter+j)
        upper_bound = max(0.0, ubi[j])
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=upper_bound,  name=var_name)
        var_list.append(var)

    if feas_act is not None:
        feas_act_post = np.maximum(feas_act, 0.)
        for j, var in enumerate(var_list[-num_neurons:]):
            var.start = feas_act_post[j]

    if len(milp_encode_idx)>0:
        for i, j in enumerate(milp_encode_idx):
            var_bin = var_list[start_idx + i]

            if(ubi[j]<=0):
               expr = var_list[relu_counter+j]
               model.addConstr(expr, GRB.EQUAL, 0)
            elif(lbi[j]>=0):
               expr = var_list[relu_counter+j] - var_list[affine_counter+j]
               model.addConstr(expr, GRB.EQUAL, 0)
            else:
               # y <= x - l(1-a)
               expr = var_list[relu_counter+j] - var_list[affine_counter+j] - lbi[j] * var_bin
               model.addConstr(expr, GRB.LESS_EQUAL, -lbi[j])

               # y >= x
               expr = var_list[relu_counter+j] - var_list[affine_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)

               # y <= u . a
               expr = var_list[relu_counter+j] - ubi[j] * var_bin
               model.addConstr(expr, GRB.LESS_EQUAL, 0)

               # y >= 0
               expr = var_list[relu_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)

               # indicator constraint
               model.addGenConstrIndicator(var_bin, True, var_list[affine_counter+j], GRB.GREATER_EQUAL, 0.0)

    if len(relax_encode_idx) > 0:
        for j in relax_encode_idx:
            if ubi[j] <= 0:
                expr = var_list[relu_counter+j]
                model.addConstr(expr, GRB.EQUAL, 0)
            elif lbi[j] >= 0:
                expr = var_list[relu_counter+j] - var_list[affine_counter+j]
                model.addConstr(expr, GRB.EQUAL, 0)

    if len(relu_groupsi) > 0:
        _add_kactivation_constraints(model, var_list, relu_groupsi, affine_counter, relu_counter)
    else:
        for j in relax_encode_idx:
            if (lbi[j] < 0) and (ubi[j] > 0):
                expr = -ubi[j] * var_list[affine_counter+j]
                expr += (ubi[j] - lbi[j]) * var_list[relu_counter+j]
                expr += lbi[j] * ubi[j]
                model.addConstr(expr, GRB.LESS_EQUAL, 0)

    return relu_counter



def handle_sign(model,var_list, affine_counter, num_neurons, lbi, ubi):
    start_idx = len(var_list)
    binary_counter = start_idx
    sign_counter = start_idx + num_neurons
    for j in range(num_neurons):
        var_name = "x" + str(start_idx + j)
        var = model.addVar(vtype=GRB.BINARY, name=var_name)
        var_list.append(var)

    # sign variables
    for j in range(num_neurons):
        var_name = "x" + str(sign_counter+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = 0.0, ub=1.0,  name=var_name)
        var_list.append(var)


    for j in range(num_neurons):
        if(ubi[j]<=0):
            expr = var_list[sign_counter+j]
            model.addConstr(expr, GRB.EQUAL, 0)
        elif(lbi[j]>=0):
            expr = var_list[sign_counter+j] - var_list[affine_counter+j]
            model.addConstr(expr, GRB.EQUAL, 1)
        else:
            # x >= l(1-a)
            expr = - var_list[affine_counter+j] - lbi[j]*var_list[binary_counter+j]
            model.addConstr(expr, GRB.LESS_EQUAL, -lbi[j])


            # x <= u.a
            expr = var_list[affine_counter+j] - ubi[j]*var_list[binary_counter+j]
            model.addConstr(expr, GRB.LESS_EQUAL, 0)

            # y = a
            expr = var_list[sign_counter+j]
            model.addConstr(expr, GRB.GREATER_EQUAL, var_list[binary_counter+j])

            # indicator constraint
            model.addGenConstrIndicator(var_list[binary_counter+j], True, var_list[affine_counter+j], GRB.GREATER_EQUAL, 0.0)

    return sign_counter

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def handle_tanh_sigmoid(model, var_list, affine_counter, num_neurons, lbi, ubi,
                        constraint_groups, activation_type):
    assert activation_type in ["Tanh", "Sigmoid"]
    y_counter = len(var_list)

    for j in range(num_neurons):
        var_name = "x" + str(y_counter + j)
        x_lb = lbi[j]
        x_ub = ubi[j]
        if activation_type == "Tanh":
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=math.tanh(x_lb), ub=math.tanh(x_ub), name=var_name)
        else:
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=sigmoid(x_lb), ub=sigmoid(x_ub), name=var_name)
        var_list.append(var)

    _add_kactivation_constraints(model, var_list, constraint_groups, affine_counter, y_counter)

    return y_counter


def create_model(nn, LB_N0, UB_N0, nlb, nub, relu_groups, numlayer, use_milp, is_nchw=False, partial_milp=0,
                 max_milp_neurons=-1, gpu_model=False, feas_act=None):
    model = Model("milp")
    model.setParam("OutputFlag",0)
    model.setParam(GRB.Param.FeasibilityTol, 2e-5)

    milp_activation_layers = np.nonzero([l in ["ReLU", "Maxpool"] for l in nn.layertypes])[0]

    ### Determine whcich layers, if any to encode with MILP
    if partial_milp < 0:
        partial_milp = len(milp_activation_layers)
    first_milp_layer = len(nn.layertypes) if partial_milp == 0 else milp_activation_layers[-min(partial_milp, len(milp_activation_layers))]

    num_pixels = len(LB_N0)

    ### Set layer counters to 0, later used to access layer resources
    ffn_counter = 0
    conv_counter = 0
    residual_counter = 0
    pool_counter = 0
    pad_counter = 0
    activation_counter = 0
    bias_counter = 0

    var_list = []
    counter = 0

    ### Encode inputs, either from box or zonotope
    if len(UB_N0)==0:
        ### Zonotope
        num_pixels = nn.zonotope.shape[0]
        num_error_terms = nn.zonotope.shape[1]
        for j in range(num_error_terms-1):
            var_name = "e" + str(j)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=var_name)
            var_list.append(var)
        counter = num_error_terms-1
        for i in range(num_pixels):
            lower_bound = nn.zonotope[i][0]
            upper_bound = lower_bound
            for j in range(1,num_error_terms):
                lower_bound = lower_bound - abs(nn.zonotope[i][j])
                upper_bound = upper_bound + abs(nn.zonotope[i][j])
            var_name = "x" + str(i)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound, ub=upper_bound, name=var_name)
            var_list.append(var)
            expr = LinExpr()
            expr += -1 * var_list[counter + i]
            for j in range(num_error_terms-1):
                expr.addTerms(nn.zonotope[i][j+1], var_list[j])

            expr.addConstant(nn.zonotope[i][0])
            model.addConstr(expr, GRB.EQUAL, 0)
    else:
        for i in range(num_pixels):
            var_name = "x" + str(i)
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_N0[i], ub=UB_N0[i], name=var_name)
            var_list.append(var)

    start_counter = [counter]
    ### Add layers to model one by one
    for i in range(numlayer):
        feas_act_i = None if feas_act is None else feas_act[i]
        if nn.layertypes[i] in ['SkipCat']:
            continue
        elif nn.layertypes[i] in ['FC']:
            weights = nn.weights[ffn_counter]
            biases = nn.biases[ffn_counter + conv_counter + bias_counter]
            index = nn.predecessors[i+1][0]
            #print("index ", index, start_counter,i, len(relu_groups))
            counter = start_counter[index]
            counter = handle_affine(model, var_list, counter, weights, biases, nlb[i], nub[i], feas_act=feas_act_i)
            ffn_counter += 1
            start_counter.append(counter)

        elif nn.layertypes[i] in ['Bias']:
            biases = nn.biases[ffn_counter+conv_counter+bias_counter]
            index = nn.predecessors[i+1][0]
            #print("index ", index, start_counter,i, len(relu_groups))
            counter = start_counter[index]
            counter = handle_bias(model, var_list, counter, biases, nlb[i], nub[i], feas_act=feas_act_i)
            bias_counter += 1
            start_counter.append(counter)

        elif(nn.layertypes[i]=='ReLU'):
            index = nn.predecessors[i+1][0]
            partial_milp_neurons = (first_milp_layer <= i) * (max_milp_neurons if max_milp_neurons >= 0 else len(nlb[i]))
            if relu_groups is None or len(relu_groups)==0:
                counter = handle_relu(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1], [], use_milp, partial_milp_neurons, feas_act=None if feas_act is None else np.array(feas_act[index-1]))
            else:
                counter = handle_relu(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1], relu_groups[activation_counter], use_milp, partial_milp_neurons, feas_act=None if feas_act is None else np.array(feas_act[index-1]))
            activation_counter += 1
            start_counter.append(counter)

        elif nn.layertypes[i] == 'Sigmoid' or nn.layertypes[i] == 'Tanh':
            index = nn.predecessors[i+1][0]
            if relu_groups is None:
                counter = handle_tanh_sigmoid(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1],
                                          [], nn.layertypes[i])
            else:
                counter = handle_tanh_sigmoid(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1],
                                          relu_groups[activation_counter], nn.layertypes[i])
            activation_counter += 1
            start_counter.append(counter)
            
        elif nn.layertypes[i] == 'Sign':
            index = nn.predecessors[i+1][0]
            counter = handle_sign(model, var_list, counter, len(nlb[i]), nlb[index-1], nub[index-1])
            activation_counter += 1
            start_counter.append(counter)

        elif nn.layertypes[i] in ['Conv']:
            filters = nn.filters[conv_counter]
            biases = nn.biases[ffn_counter + conv_counter + bias_counter]
            filter_size = nn.filter_size[conv_counter]
            out_shape = nn.out_shapes[conv_counter + pool_counter + pad_counter]
            padding = nn.padding[conv_counter + pool_counter + pad_counter]
            strides = nn.strides[conv_counter + pool_counter]
            input_shape = nn.input_shape[conv_counter + pool_counter + pad_counter]
            index = nn.predecessors[i+1][0]
            counter = start_counter[index]
            counter = handle_conv(model, var_list, counter, filters, biases, filter_size, input_shape, strides,
                                  out_shape, padding[0], padding[1], padding[2], padding[3], nlb[i], nub[i],
                                  is_nchw=is_nchw, feas_act=feas_act_i)
            start_counter.append(counter)
            conv_counter += 1

        elif(nn.layertypes[i] in ['Maxpool',"MaxPool"]):
            partial_milp_neurons = (first_milp_layer <= i) * (max_milp_neurons if max_milp_neurons >= 0 else len(nlb[i]))
            pool_size = nn.pool_size[pool_counter]
            input_shape = nn.input_shape[conv_counter + pool_counter + pad_counter]
            out_shape = nn.out_shapes[conv_counter + pool_counter + pad_counter]
            padding = nn.padding[conv_counter + pool_counter + pad_counter]
            strides = nn.strides[conv_counter + pool_counter]
            index = nn.predecessors[i+1][0]
            counter = start_counter[index]
            counter = handle_maxpool(model, var_list, i, counter, pool_size, input_shape, strides, out_shape, padding[0],
                                     padding[1], nlb[i], nub[i], nlb[i-1], nub[i-1], use_milp, is_nchw=is_nchw, feas_act=feas_act_i)
            start_counter.append(counter)
            pool_counter += 1

        elif(nn.layertypes[i]=='Pad'):
            input_shape = nn.input_shape[conv_counter + pool_counter + pad_counter]
            out_shape = nn.out_shapes[conv_counter + pool_counter + pad_counter]
            padding = nn.padding[conv_counter + pool_counter + pad_counter]
            index = nn.predecessors[i+1][0]
            counter = start_counter[index]
            counter = handle_padding(model, var_list, counter, input_shape, out_shape, padding[0], padding[1],
                                     padding[2], padding[3], nlb[i], nub[i], is_nchw=is_nchw, feas_act=feas_act_i)
            start_counter.append(counter)
            pad_counter += 1

        elif nn.layertypes[i] in ['Resadd']:
            index1 = nn.predecessors[i+1][0]
            index2 = nn.predecessors[i+1][1]
            counter1 = start_counter[index1]
            counter2 = start_counter[index2]
            counter = handle_residual(model, var_list, counter1, counter2, nlb[i], nub[i], feas_act=feas_act_i)
            start_counter.append(counter)
            residual_counter += 1

        # elif nn.layertypes[i] in ['Pad']:
        #     raise NotImplementedError
        else:
            assert 0, 'layertype:' + nn.layertypes[i] + 'not supported for refine'

    #model.write("model_refinepoly.lp")
    np.set_printoptions(threshold=sys.maxsize)
    #print("NLB ", nlb[-4], len(nlb[-4]))
    return counter, var_list, model


class Cache:
    model = None
    output_counter = None
    lbi = None
    ubi = None


def solver_call(ind):
    model = Cache.model.copy()
    runtime = 0

    obj = LinExpr()
    obj += model.getVars()[Cache.output_counter+ind]
    #print (f"{ind} {model.getVars()[Cache.output_counter+ind].VarName}")

    model.setObjective(obj, GRB.MINIMIZE)
    model.reset()
    model.optimize()
    runtime += model.RunTime
    soll = Cache.lbi[ind] if model.SolCount==0 else model.objbound
    # print (f"{ind} {model.status} lb ({Cache.lbi[ind]}, {soll}) {model.RunTime}s")
    sys.stdout.flush()

    model.setObjective(obj, GRB.MAXIMIZE)
    model.reset()
    model.optimize()
    runtime += model.RunTime
    solu = Cache.ubi[ind] if model.SolCount==0 else model.objbound
    # print (f"{ind} {model.status} ub ({Cache.ubi[ind]}, {solu}) {model.RunTime}s")
    sys.stdout.flush()

    soll = max(soll, Cache.lbi[ind])
    solu = min(solu, Cache.ubi[ind])

    addtoindices = (soll > Cache.lbi[ind]) or (solu < Cache.ubi[ind])

    return soll, solu, addtoindices, runtime


def get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, output_size, nlb, nub, relu_groups,
                                   use_milp, candidate_vars, timeout, is_nchw, gpu_model=False, start_time=None, feas_act=None):
    lbi = nlb[abs_layer_count]
    ubi = nub[abs_layer_count]

    widths = [u-l for u, l in zip(ubi, lbi)]
    candidate_vars = sorted(candidate_vars, key=lambda k: widths[k])
    # if start_time is not None and config.timeout_complete is not None:
    #     candidate_vars = candidate_vars[:int(max(0, np.floor((start_time + config.timeout_complete * 0.8 - time.time())/timeout)))]
    #     if len(candidate_vars) == 0:
    #         return lbi, ubi, []

    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, relu_groups, layerno+1, use_milp,
                                            partial_milp=-1 if use_milp else 0, max_milp_neurons=-1, gpu_model=gpu_model,
                                            is_nchw=is_nchw, feas_act=feas_act)
    resl = [0]*len(lbi)
    resu = [0]*len(ubi)
    indices = []

    model.setParam(GRB.Param.TimeLimit, timeout)
    model.setParam(GRB.Param.Threads, 2)
    output_counter = counter

    model.update()
    model.reset()

    NUMPROCESSES = config.numproc
    Cache.model = model
    Cache.output_counter = output_counter
    Cache.lbi = lbi
    Cache.ubi = ubi
    Cache.time_limit = timeout

    refined = [False]*len(lbi)

    for v in candidate_vars:
        refined[v] = True
        #print (f"{v} {deltas[v]} {widths[v]} {deltas[v]/widths[v]}")
    with multiprocessing.Pool(NUMPROCESSES) as pool:
        solver_result = pool.map(solver_call, candidate_vars)

    for (l, u, addtoindices, runtime), ind in zip(solver_result, candidate_vars):
        resl[ind] = l
        resu[ind] = u

        if (l > u):
            print(f"unsound {ind}")

        if addtoindices:
            indices.append(ind)

    for i, flag in enumerate(refined):
        if not flag:
            resl[i] = lbi[i]
            resu[i] = ubi[i]

    for i in range(abs_layer_count):
        for j in range(len(nlb[i])):
            if(nlb[i][j]>nub[i][j]):
                print("fp unsoundness detected ", nlb[i][j],nub[i][j],i,j)

    return resl, resu, sorted(indices)


def add_spatial_constraints(model, spatial_constraints, var_list, input_size):

    delta = spatial_constraints.get('delta')
    gamma = spatial_constraints.get('gamma')
    channels = spatial_constraints.get('channels')

    lower_planes = spatial_constraints.get('lower_planes')
    upper_planes = spatial_constraints.get('upper_planes')

    add_norm_constraints = spatial_constraints.get('add_norm_constraints')
    neighboring_indices = spatial_constraints.get('neighboring_indices')

    vector_field = list()

    for idx in range(input_size // channels):
        vx = model.addVar(lb=-delta, ub=delta)
        vy = model.addVar(lb=-delta, ub=delta)
        add_norm_constraints(model, vx, vy)
        vector_field.append({'vx': vx, 'vy': vy})

    if (lower_planes is not None) and (upper_planes is not None):

        for idx, vector in enumerate(vector_field):
            for channel in range(channels):
                index = channels * idx + channel
                var = var_list[index]
                lb_a, ub_a = lower_planes[0][index], upper_planes[0][index]
                lb_b, ub_b = lower_planes[1][index], upper_planes[1][index]
                lb_c, ub_c = lower_planes[2][index], upper_planes[2][index]

                model.addConstr(
                    var >= lb_a + lb_b * vector['vx'] + lb_c * vector['vy']
                )
                model.addConstr(
                    var <= ub_a + ub_b * vector['vx'] + ub_c * vector['vy']
                )

    if (gamma is not None) and (gamma < float('inf')):

        indices = neighboring_indices['indices'][::channels] // channels
        neighbors = neighboring_indices['neighbors'][::channels] // channels

        for idx, nbr in zip(indices.tolist(), neighbors.tolist()):
            model.addConstr(
                vector_field[idx]['vx'] - vector_field[nbr]['vx'] <= gamma
            )
            model.addConstr(
                vector_field[nbr]['vx'] - vector_field[idx]['vx'] <= gamma
            )
            model.addConstr(
                vector_field[idx]['vy'] - vector_field[nbr]['vy'] <= gamma
            )
            model.addConstr(
                vector_field[nbr]['vy'] - vector_field[idx]['vy'] <= gamma
            )


def verify_network_with_milp(config, nn, LB_N0, UB_N0, nlb, nub, constraints, spatial_constraints=None, is_nchw=False,
                             gpu_model=False, start_time=None, find_adex=False, verbosity=2, feas_act=None):
    numlayer = nn.numlayer
    input_size = len(LB_N0)
    start_milp = time.time()
    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, None, numlayer, use_milp=True, is_nchw=is_nchw,
                                            partial_milp=-1, max_milp_neurons=-1, gpu_model=gpu_model, feas_act=feas_act)
    #print("timeout ", config.timeout_milp)

    if spatial_constraints is not None:
        add_spatial_constraints(model, spatial_constraints, var_list, input_size)
                
    adv_examples = []
    non_adv_examples = []
    adv_val = []
    non_adv_val = []
    for or_list in constraints:
        or_result = False
        for is_greater_tuple in or_list:
            obj = obj_from_is_greater_tuple(is_greater_tuple, var_list, counter)
            model.setObjective(obj, GRB.MINIMIZE)

            # In some cases it occurs that Gurobi reports an infeasible model
            # probably due to numerical difficulties (c.f. https://github.com/eth-sri/eran/issues/74).
            # These can be resolved (in the cases considered) by increasing the Cutoff parameter.
            # The code below tries to recover from an infeasible model by increasing the default cutoff
            # a few times.
            # 0.01 is the default cutoff value
            for cutoff in [0.01, 0.1, GRB.INFINITY]:
                model.reset()
                milp_timeout = config.timeout_final_milp if config.timeout_complete is None else (config.timeout_complete + start_milp - time.time())
                model.setParam(GRB.Param.TimeLimit, milp_timeout)
                model.setParam(GRB.Param.Cutoff, cutoff)
                if find_adex:
                    model.optimize(milp_callback_adex)
                else:
                    model.optimize(milp_callback)
                if model.status not in [3, 4]:  # status 3 and 4 indicate an infeasible model
                    # no infeasibility reported.
                    break
                else:
                    warnings.warn("Infeasible model encountered. Trying to increase the Cutoff parameter to recover.")
            else:
                # all values led to an infeasible model
                assert model.status not in [3, 4], f"Infeasible model encountered. Model status {model.status}"

            obj_bound = f"{model.objbound:.4f}" if hasattr(model, "objbound") else "failed"
            obj_val = f"{model.objval:.4f}" if hasattr(model, "objval") else "failed"
            if verbosity >= 2:
                print(f"MILP model status: {model.Status}, Obj val/bound for constraint {is_greater_tuple}: {obj_val}/{obj_bound}, Final solve time: {model.Runtime:.3f}")

            if model.Status in [1]:
                pass #model loaded but not run ==> out of time
            elif model.Status == 6 or model.objbound > 0:
                or_result = True
                if model.solcount > 0:
                    non_adv_examples.append(model.x[0:input_size])
                    non_adv_val.append(model.objval)
                break
            elif model.solcount > 0:
                adv_examples.append(model.x[0:input_size])
                adv_val.append(model.objval)

        if not or_result and (config.timeout_complete + start_time - time.time()) > 0.1:
            # Per default, we try to show that one of the elements of the or_list holds for the whole domain this might
            # not be the case, even if the property holds. To find potential counterexamples, we add the negation of
            # all or-clauses to the model and solve the corresponding feasibility problem.
            # If this returns an infeasible model we allow this to be used for certification by setting
            # certify_with_feasibility to True.
            if len(or_list) > 1:
                model.NumObj = 0
                for i, is_greater_tuple in enumerate(or_list):
                    obj_constr = obj_from_is_greater_tuple(is_greater_tuple, var_list, counter)
                    model.addConstr(obj_constr, GRB.LESS_EQUAL, 0, name=f"Adex_Obj_{i:d}")
                model.setParam(GRB.Param.TimeLimit, config.timeout_final_milp)
                model.setParam(GRB.Param.Cutoff, GRB.INFINITY)
                model.setParam(GRB.Param.FeasibilityTol, 5e-5)
                model.reset()
                model.optimize()
                sol_count = f"{model.solcount:d}" if hasattr(model, "solcount") else "None"
                if verbosity >= 1:
                    print(f"MILP adex model status: {model.Status}, Model solution count: {sol_count}, Final solve time: {model.Runtime:.3f}")
                if model.solcount > 0:
                    # This yields a guaranteed adversarial example
                    adv_examples = [model.x[0:input_size]]
                    adv_val = [None]
                # The below portion enables the use of feasibility instead of optimization based  certification.
                # This is not recommended as GUROBI is known to sometimes return spurious infeasibility
                certify_with_feasibility = False
                if certify_with_feasibility and model.status in [3, 4]: # model is infeasible
                    for i in range(len(or_list)):
                        model.remove(model.getConstrByName(f"Adex_Obj_{i:d}"))
                    model.reset()
                    model.optimize()
                    if model.status not in [3, 4]: # not infeasible anymore
                        print(f"MILP adex model status without adex constraints: {model.Status}, Final solve time: {model.Runtime:.3f}")
                        warnings.warn("Model feasibility used for certification.")
                        or_result = True
        if not or_result:
            if len(adv_examples) > 0:
                return False, adv_examples, adv_val
            else:
                return False, None, None
    if len(non_adv_examples) > 0:
        return True, non_adv_examples, non_adv_val
    else:
        return True, None, None


def refine_input_with_constr(model, var_list, counter, lbi, ubi, or_list, start_time, time_out_ind, time_out_total, input_idxs=None):
    # model = model.copy()
    for i, is_greater_tuple in enumerate(or_list):
        obj_constr = obj_from_is_greater_tuple(is_greater_tuple, var_list, counter)
        model.addConstr(obj_constr, GRB.LESS_EQUAL, 0, name=f"Adex_Obj_{i:d}")
    runtime = 0

    lbi_new = lbi.copy()
    ubi_new = ubi.copy()

    old_cutoff = model.getParamInfo(GRB.Param.Cutoff)[2]

    if input_idxs is None:
        input_idxs = range(len(ubi))

    for idx in input_idxs:
        if check_timeleft(config, start_time, min(time_out_ind,time_out_total-runtime))<1:
            break
        obj = LinExpr()
        obj += var_list[idx]
        model.setObjective(obj, GRB.MINIMIZE)
        model.setParam(GRB.Param.Cutoff, GRB.INFINITY)
        model.setParam(GRB.Param.TimeLimit, check_timeleft(config, start_time, min(time_out_ind,time_out_total-runtime)))
        model.reset()
        model.optimize()
        if hasattr(model, "objbound"):
            lbi_new[idx] = model.objbound
        elif model.status==2 and hasattr(model, "objval"):
            lbi_new[idx] = model.objval
        elif model.status==3:
            break
        else:
            lbi_new[idx] = lbi[idx]
        runtime += model.RunTime

        model.setParam(GRB.Param.TimeLimit, check_timeleft(config, start_time, min(time_out_ind, time_out_total-runtime)))
        model.setObjective(obj, GRB.MAXIMIZE)
        model.setParam(GRB.Param.Cutoff, -GRB.INFINITY)
        model.reset()
        model.optimize()
        runtime += model.RunTime
        if hasattr(model, "objbound"):
            ubi_new[idx] = model.objbound
        elif model.status==2 and hasattr(model, "objval"):
            ubi_new[idx] = model.objval
        elif model.status==3:
            break
        else:
            ubi_new[idx] = ubi[idx]
        # print (f"{ind} {model.status} ub ({Cache.ubi[ind]}, {solu}) {model.RunTime}s")
        # sys.stdout.flush()
    if model.status==3:
        for i, is_greater_tuple in enumerate(or_list):
            model.remove(model.getConstrByName(f"Adex_Obj_{i:d}"))
        model.reset()
        model.optimize()
        if model.status==3:
            assert False, "model remained infeasible"
        else:
            model.reset()
            return None, None
    model.setParam(GRB.Param.Cutoff, old_cutoff)
    for i, is_greater_tuple in enumerate(or_list):
        model.remove(model.getConstrByName(f"Adex_Obj_{i:d}"))
    model.reset()

    lbi_new = np.maximum(lbi,lbi_new)
    ubi_new = np.minimum(ubi, ubi_new)
    return lbi_new, ubi_new

def evaluate_model(model, var_list, counter, input_len, is_greater_tuple, start_time, time_out, verbosity=2, callback=None, dtype=np.float64):
    or_result = False
    sol = None
    model.reset()
    obj = obj_from_is_greater_tuple(is_greater_tuple, var_list, counter)
    model.setObjective(obj, GRB.MINIMIZE)

    model.setParam(GRB.Param.TimeLimit, time_out)
    if callback is None:
        model.optimize()
    else:
        model.optimize(callback)

    obj_bound_num = model.objbound if hasattr(model, "objbound") else None

    obj_bound = f"{model.objbound:.4f}" if hasattr(model, "objbound") else "failed"
    obj_val = f"{model.objval:.4f}" if hasattr(model, "objval") else "failed"
    if verbosity >= 2:
        print(f"Model status: {model.Status}, Obj val/bound for constraint {is_greater_tuple}: {obj_val}/{obj_bound}, Final solve time: {model.Runtime:.3f}")

    if model.SolCount > 0:
        sol = np.array(model.x[0:input_len]).astype(dtype=dtype)

    if model.Status == 6:
        # Cutoff active
        or_result = True
    elif model.Status in [2, 9, 11] and obj_bound_num is not None:
        # positive objbound => sound against adv_label
        or_result = obj_bound_num > 0
    elif model.Status not in [2, 6, 9, 11]:
        if verbosity >= 1:
            print("Model was not successful status is", model.Status)
        assert model.Status not in [3, 4], "Infeasible model encountered"

    return or_result, sol, obj_bound_num


def evaluate_models(model_lp, var_list_lp, counter_lp, input_len, constraints, terminate_on_failure, model_p_milp=None,
                    var_list_p_milp=None, counter_p_milp=None, eval_instance=None, find_adex=False, verbosity=2,
                    start_time=None, dtype=np.float64):
    ### Evaluate an LP and partial MILP model for one of the refine domains

    adex_list = []
    and_result = True
    updated_constraint = constraints.copy()
    model_bounds = {}
    adex_found = False

    for _ in range(len(constraints)):
        if adex_found: break
        or_list = updated_constraint.pop(0)
        # OR-Constraint
        or_result = False
        ### first do standard lp for all elements in or list
        for is_greater_tuple in or_list:
            if adex_found: break

            or_result, adex, obj_bound = evaluate_model(model_lp, var_list_lp, counter_lp, input_len, is_greater_tuple, start_time, config.timeout_final_lp, verbosity, dtype=dtype)
            model_bounds[is_greater_tuple] = (obj_bound)

            if adex is not None:
                if eval_instance is not None:
                    cex_label, cex_out = eval_instance(adex)
                    adex_found = not evaluate_cstr(constraints, cex_out)
                if eval_instance is None or adex_found:
                    adex_list.append(adex)
            if adex_found or or_result:
                break

        ### try and find adex with lp
        if find_adex and not or_result and not adex_found and len(constraints)==1 and len(constraints[0])>1 and eval_instance is not None:
            _, adex, _ = evaluate_model(model_lp, var_list_lp, counter_lp, input_len, constraints[0], start_time, config.timeout_final_lp, verbosity, dtype=dtype)
            if adex is not None:
                if eval_instance is not None:
                    cex_label, cex_out = eval_instance(adex)
                    adex_found = not evaluate_cstr(constraints, cex_out)
                if eval_instance is None or adex_found:
                    adex_list.append(adex)

        ### do partial milp for elements in or list
        if model_p_milp is not None and (not or_result) and (not adex_found):
            for is_greater_tuple in or_list:
                or_result, adex, obj_bound = evaluate_model(model_p_milp, var_list_p_milp, counter_p_milp, input_len,
                                                            is_greater_tuple, start_time, config.timeout_final_milp,
                                                            verbosity, callback=(milp_callback_adex if find_adex else milp_callback), dtype=dtype)
                if obj_bound is not None:
                    model_bounds[is_greater_tuple] = float(obj_bound) if model_bounds[is_greater_tuple] is None else np.maximum(model_bounds[is_greater_tuple], float(obj_bound))

                if adex is not None:
                    if eval_instance is not None:
                        cex_label, cex_out = eval_instance(adex)
                        adex_found = not evaluate_cstr(constraints, cex_out)
                    if eval_instance is None or adex_found:
                        adex_list.append(adex)
                if adex_found or or_result:
                    break

        if not or_result:
            and_result = False
            updated_constraint.append(or_list)
            if terminate_on_failure or adex_found:
                break

    return and_result, updated_constraint, adex_list, model_bounds


def obj_from_is_greater_tuple(is_greater_tuple, var_list, counter):
    ### Define an objective for GUROBI based on an is_greater tuple
    obj = LinExpr()
    if not isinstance(is_greater_tuple,list):
        is_greater_tuple = [is_greater_tuple]

    for igt in is_greater_tuple:
        (i, j, k) = igt
        if i == -1:  # var[i] > k
            obj += 1 * var_list[counter + j] - float(k)
        elif j == -1:  # var[j] < k
            obj += float(k) - 1 * var_list[counter + i]
        elif i != j:  # var[i] > var[j]
            obj += 1 * var_list[counter + i]
            obj += -1 * var_list[counter + j]
        else:
            assert False, f"invalid constraint encountered {is_greater_tuple}"
    return obj
    ### TODO make logic of constraints with constant component consistent