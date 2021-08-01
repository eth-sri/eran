import re
import numpy as np
import copy
import torch
import time
# from eran import ERAN
# from convert_nets import read_onnx_net, create_torch_net
import pickle as pkl
import os

def identify_var(var_string):
    match = re.match("([\-,\+]*)([A-Z,a-z,_]+)_([0-9]*)", var_string)
    if match is None:
        assert False, var_string
    else:
        var_group = match.group(2)
        var_idx = int(match.group(3))
    return var_group, var_idx


def check_numeric(var_string, dtype=np.float32):
    match = re.match("([\-,\+]*)([0-9]*(\.[0-9]*)?(e[\-,\+]?[0-9]+)?)", var_string)
    var_string = "".join((var_string).split())
    if match is None or len(match.group(2))==0 or match.group(2) is None:
        return None
    else:
        # sign = -1 if match.group(1)=="-" else 1
        try:
            num = dtype(var_string)
            return num
        except:
            assert False, f"Could not translate numeric string {var_string}"


def extract_terms(input_term, dtype=np.float32):
    terms = [term.strip() for term in input_term.split(" ")]
    sign_flag = None
    output_terms = []
    for term in terms:
        if term == "": continue
        if term == "-":
            sign_flag = -1
        elif term == "+":
            sign_flag = +1
        else:
            num = check_numeric(term, dtype)
            if num is None:
                if term.startswith("-"):
                    sign_flag = -1 if sign_flag is None else -1 * sign_flag
                elif term.startswith("+"):
                    sign_flag = +1
                var_group, var_idx = identify_var(term)
                value = 1 if sign_flag is None else sign_flag
            else:
                var_group = "const"
                var_idx = -1
                value = num
            output_terms.append((var_group, var_idx, value))
            sign_flag = None
    return output_terms


def identify_variables(lines):
    net_inputs = []
    net_outputs = []

    for line in lines:
        if line.startswith(";"): continue
        if line.startswith("(declare-const"):
            match = re.match("\(declare-const ([A-Z,a-z,_]+)_([0-9]*) ([A-Z,a-z]*)\)", line)
            if match is None:
                assert False, line
            else:
                var_group = match.group(1)
                var_idx = int(match.group(2))
                var_type = match.group(3)
                if var_group == "X":
                    net_inputs.append(("X", var_idx, var_type))
                elif var_group == "Y":
                    net_outputs.append(("Y", var_idx, var_type))
                else:
                    assert False, f"Unrecognized variable:\n{line}"
    return net_inputs, net_outputs


def parse_vnn_lib_prop(file_path, dtype=np.float32):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Get all defined variables
    net_inputs, net_outputs = identify_variables(lines)

    # Input constraints of the form net_inputs >=/<= C [spec_anchors, spec_utility, 1]
    C_lb_list = [-np.ones((len(net_inputs)), dtype) * np.inf]
    C_ub_list = [np.ones((len(net_inputs)), dtype) * np.inf]

    # Output constraints of the form C [net_outputs, 1] >= 0
    C_out_list = [np.zeros((0, len(net_outputs) + 1), dtype)]

    # Dictionaries associating variables with indicies
    idx_dict = {f"{x[0]}_{x[1]}": i for i, x in enumerate(sorted(net_inputs))}
    idx_dict["const_-1"] = -1
    idx_dict_out = {f"{x[0]}_{x[1]}": i for i, x in enumerate(sorted(net_outputs))}
    idx_dict_out["const_-1"] = -1

    # Extract all constraints
    open_brackets_n = 0
    block = []
    for line in lines:
        if line.startswith(";"): continue
        if open_brackets_n == 0:
            if line.startswith("(assert"):
                open_brackets_n = 0
                open_brackets_n += line.count("(")
                open_brackets_n -= line.count(")")
                block.append(line.strip())
        else:
            open_brackets_n += line.count("(")
            open_brackets_n -= line.count(")")
            block.append(line.strip())
        if open_brackets_n == 0 and len(block) > 0:
            block = " ".join(block)
            match = re.match("\(assert(.*)\)$", block)
            C_lb_list, C_ub_list, C_out_list = parse_assert_block(match.group(1).strip(), C_lb_list, C_ub_list,
                                                                  C_out_list, idx_dict, idx_dict_out, dtype)
            block = []

    boxes, GT_constraints = translate_output_constraints(C_lb_list, C_ub_list, C_out_list)
    return boxes, GT_constraints


def parse_assert_block(block, C_lb_list, C_ub_list, C_out_list, idx_dict, idx_dict_out, dtype=np.float32):
    match = re.match("\((or|and|[>,<,=]+)(.*)\)$",block)
    if match is None:
        assert False, block
    else:
        spec_relation = match.group(1)
        spec_content = match.group(2)
        if spec_relation in ["or", "and"]:
            if spec_relation == "or":
                C_lb_list_new = []
                C_ub_list_new = []
                C_out_list_new = []
            open_brackets_n = 0
            mini_block = []
            for c in spec_content:
                if c =="(":
                    open_brackets_n += 1
                    mini_block.append(c)
                elif open_brackets_n > 0:
                    mini_block.append(c)
                    if c == ")":
                        open_brackets_n -= 1
                        if open_brackets_n==0:
                            mini_block =  "".join(mini_block).strip()
                            if spec_relation == "or":
                                C_lb_list_tmp, C_ub_list_tmp, C_out_list_tmp = parse_assert_block(mini_block,
                                                                                   copy.deepcopy(C_lb_list), copy.deepcopy(C_ub_list),
                                                                                   copy.deepcopy(C_out_list), idx_dict, idx_dict_out, dtype)
                                C_lb_list_new += C_lb_list_tmp
                                C_ub_list_new += C_ub_list_tmp
                                C_out_list_new += C_out_list_tmp
                            elif spec_relation == "and":
                                C_lb_list, C_ub_list, C_out_list = parse_assert_block(mini_block, C_lb_list, C_ub_list,
                                                                                      C_out_list, idx_dict, idx_dict_out, dtype)
                            mini_block = []
            assert open_brackets_n == 0
            if spec_relation == "or":
                C_lb_list, C_ub_list, C_out_list = C_lb_list_new, C_ub_list_new, C_out_list_new
        else:
            n_y = C_out_list[0].shape[-1]-1
            var_idx, c_lb, c_ub, c_out = parse_assert_content(spec_content.strip(), spec_relation.strip(), n_y, idx_dict_out, dtype)
            C_lb_list, C_ub_list, C_out_list = add_constraints(var_idx, c_lb, c_ub, c_out, C_lb_list, C_ub_list, C_out_list, idx_dict)
    return C_lb_list, C_ub_list, C_out_list


def parse_assert_content(spec_content, spec_relation, n_y, idx_dict_out, dtype=np.float32):
    c_lb, c_ub, c_out = None, None, None
    if spec_content.startswith("("):
        match = re.match("\((.*?)\) .*",spec_content)
        if match is None:
            assert False, spec_content
        else:
            first_term = match.group(1)
    else:
        match = re.match("^([A-Z,a-z,_,\-,\+,0-9,\.]*).*?",spec_content)
        if match is None:
            assert False, spec_content
        else:
            first_term = match.group(1)
    # get second term of constraint
    if spec_content.endswith(")"):
        match = re.match(".*\((.*)\)",spec_content)
        if match is None:
            assert False, spec_content
        else:
            second_term = match.group(1)
    else:
        match = re.match(".*?([A-Z,a-z,_,\.,\-,\+,0-9]*)$",spec_content)
        if match is None:
            assert False, spec_content
        else:
            second_term = match.group(1)
    assert spec_relation in [">=","<="]
    g_terms = extract_terms(first_term if spec_relation==">=" else second_term, dtype)
    l_terms = extract_terms(second_term if spec_relation==">=" else first_term, dtype)
    # Input Constraints
    if (len(g_terms)==1 and g_terms[0][0]=="X"):
        # lower bound on input
        var_idx = f"{g_terms[0][0]}_{g_terms[0][1]}"
        assert len(l_terms)==1 and l_terms[0][0]=="const", "only box constraints are supported for the input"
        c_lb = l_terms[0][2]
    elif (len(l_terms)==1 and l_terms[0][0]=="X"):
        # upper bound on input
        var_idx = f"{l_terms[0][0]}_{l_terms[0][1]}"
        assert len(g_terms)==1 and g_terms[0][0]=="const", "only box constraints are supported for the input"
        c_ub = g_terms[0][2]
    else:
        # Output Constraint
        c_out = np.zeros((1, n_y+1), dtype)
        var_idx = "Y"
        for term in g_terms:
            var_key = f"{term[0]}_{term[1]}"
            assert var_key in idx_dict_out
            c_out[0, idx_dict_out[var_key]] += term[2]
        for term in l_terms:
            var_key = f"{term[0]}_{term[1]}"
            assert var_key in idx_dict_out
            c_out[0, idx_dict_out[var_key]] -= term[2]
    return var_idx, c_lb, c_ub, c_out


def add_constraints(var_idx, c_lb, c_ub, c_out, C_lb_list, C_ub_list, C_out_list, idx_dict):
    C_out_list_new = []
    for C_lb, C_ub, C_out in zip(C_lb_list, C_ub_list, C_out_list):
        if c_out is not None:
            C_out = np.concatenate([C_out, c_out], axis=0)
        if c_lb is not None:
            C_lb[idx_dict[var_idx]] = max(C_lb[idx_dict[var_idx]], c_lb)
        if c_ub is not None:
            C_ub[idx_dict[var_idx]] = min(C_ub[idx_dict[var_idx]], c_ub)
        C_out_list_new.append(C_out)
    return C_lb_list, C_ub_list, C_out_list_new


def translate_output_constraints(C_lb_list, C_ub_list, C_out_list):
    # Counterexample definition of the form C [net_outputs, 1] >= 0
    unique_lb = []
    unique_ub = []
    lb_map = []
    ub_map = []

    for C_lb in C_lb_list:
        for i, C_lb_ref in enumerate(unique_lb):
            if np.isclose(C_lb_ref, C_lb).all():
                lb_map.append(i)
                break
        else:
            lb_map.append(len(unique_lb))
            unique_lb.append(C_lb)
    for C_ub in C_ub_list:
        for i, C_ub_ref in enumerate(unique_ub):
            if np.isclose(C_ub_ref, C_ub).all():
                ub_map.append(i)
                break
        else:
            ub_map.append(len(unique_ub))
            unique_ub.append(C_ub)

    spec_map = []
    specs = []
    for i_lb, i_ub in zip(lb_map, ub_map):
        for i, spec_ref in enumerate(specs):
            if spec_ref == (i_lb, i_ub):
                spec_map.append(i)
                break
        else:
            spec_map.append(len(specs))
            specs.append((i_lb, i_ub))

    boxes = [(unique_lb[specs[i_spec][0]], unique_ub[specs[i_spec][1]]) for i_spec in sorted(list(set(spec_map)))]

    C_out_specs = [[] for _ in range(len(specs))]
    for i, spec_idx in enumerate(spec_map):
        C_out_specs[spec_idx].append(C_out_list[i])  # or_list of and_arrays

    GT_specs = []
    for C_out_spec in C_out_specs:
        and_list = []
        for and_array in C_out_spec:
            or_list = []
            for i in range(and_array.shape[0]):
                numeric = and_array[i, -1]
                if numeric != 0:
                    l_label = (and_array[i, 0:-1] < 0).nonzero()[0]
                    g_label = (and_array[i, 0:-1] > 0).nonzero()[0]
                    assert len(l_label) + len(g_label) == 1
                    if len(g_label) == 1:
                        or_list.append((-1, int(g_label), -numeric/np.abs(and_array[i,g_label])[0])) # intentional negation
                    elif len(l_label) == 1:
                        or_list.append((int(l_label), -1, numeric/np.abs(and_array[i,l_label])[0])) # intentional negation
                    else:
                        assert False
                else:
                    g_label = (and_array[i, 0:-1] == -1).nonzero()[0]
                    l_label = (and_array[i, 0:-1] == 1).nonzero()[0]
                    assert len(l_label) == 1 and len(g_label) == 1
                    or_list.append((g_label[0], l_label[0], 0))
            if not or_list in and_list:
                and_list.append(or_list)
            else:
                print(f"duplicate constraint detected:",and_list,or_list)
        GT_specs.append(and_list)
    return boxes, GT_specs


def translate_box_to_sample(boxes, equal_limits=True):
    if equal_limits:
        data_lb = np.stack([x for y in boxes for x in y], axis=0).min()
        data_ub = np.stack([x for y in boxes for x in y], axis=0).max()
        eps = np.stack([y[1]-y[0] for y in boxes], axis=0).max()/2
    else:
        data_lb = np.stack([x for y in boxes for x in y],axis=0).min(axis=0)
        data_ub = np.stack([x for y in boxes for x in y], axis=0).max(axis=0)
        eps = np.stack([y[1] - y[0] for y in boxes], axis=0).max(axis=0)/2

    samples = []
    for box in boxes:
        lb, ub = box
        sample = np.where(lb == data_lb, ub - eps, np.where(ub == data_ub, lb + eps, (ub + lb) / 2))
        samples.append(np.clip(sample, lb, ub))
    return samples, eps


def translate_constraints_to_label(GT_specs):
    labels = []
    for and_list in GT_specs:
        label = None
        for or_list in and_list:
            if len(or_list)>1:
                label = None
                break
            if label is None:
                label = or_list[0][0]
            elif label != or_list[0][0]:
                label = None
                break
        labels.append(label)
    return labels


def evaluate_cstr(constraints, net_out, torch_input=False):
    if len(net_out.shape) <= 1:
        net_out = net_out.reshape(1,-1)

    n_samp = net_out.shape[0]

    and_holds = torch.ones(n_samp, dtype=bool, device=net_out.device) if torch_input else np.ones(n_samp,dtype=np.bool)
    for or_list in constraints:
        or_holds = torch.zeros(n_samp, dtype=bool, device=net_out.device) if torch_input else np.zeros(n_samp, dtype=np.bool)
        for cstr in or_list:
            if cstr[0] == -1:
                or_holds = or_holds.__or__(cstr[2] > net_out[:, cstr[1]])
            elif cstr[1] == -1:
                or_holds = or_holds.__or__(net_out[:, cstr[0]] > cstr[2])
            else:
                or_holds = or_holds.__or__(net_out[:, cstr[0]] - net_out[:, cstr[1]] > cstr[2])
            if or_holds.all():
                break
        and_holds = and_holds.__and__(or_holds)
        if not and_holds.any():
            break
    return and_holds


def negate_cstr_or_list(or_list):
    neg_and_list = []
    for is_greater_tuple in or_list:
        (i, j, k) = is_greater_tuple

        if i == -1:  # var[j] > k
            neg_and_list.append([(j,-1,k)])
        elif j == -1:  # var[i] < k
            neg_and_list.append([(-1,i,k)])
        elif i != j:  # var[i] > var[j]
            neg_and_list.append([(j,i,-k)])
        else:
            assert False, f"invalid constraint encountered {is_greater_tuple}"
    return neg_and_list


def translate_gurobi_status(status_id):
    gurobi_status_dict = {
        1: "LOADED",
        2: "Optimal",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        }
    return gurobi_status_dict[status_id]


def check_timeout(config, start_time, min_remaining=0.1):
    if config.timeout_complete is None:
        return False
    if start_time is None:
        return False
    if (config.timeout_complete + start_time - time.time()) < min_remaining:
        return True


def check_timeleft(config, start_time, alternative_timeout):
    if config.timeout_complete is None:
        return alternative_timeout
    if start_time is None:
        return alternative_timeout
    return max(0, min(alternative_timeout, start_time+config.timeout_complete-time.time()))