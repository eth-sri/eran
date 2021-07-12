import re
import numpy as np


def identify_var(var_string):
    match = re.match("([\-,\+]*)([A-Z,a-z,_]+)_([0-9]*)", var_string)
    if match is None:
        assert False, var_string
    else:
        var_group = match.group(2)
        var_idx = int(match.group(3))
    return var_group, var_idx


def check_numeric(var_string):
    match = re.match("([\-,\+]*)([0-9]*(\.[0-9]*)?)", var_string)
    if match is None or len(match.group(2))==0 or match.group(2) is None:
        return None
    else:
        sign = -1 if match.group(1)=="-" else 1
        return sign * float(match.group(2))


def extract_terms(input_term):
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
            num = check_numeric(term)
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
    spec_anchors = []
    spec_utility = []
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
                elif var_group == "X_hat":
                    spec_anchors.append(("X_hat", var_idx, var_type))
                else:
                    spec_utility.append((var_group, var_idx, var_type))

    return net_inputs, net_outputs, spec_anchors, spec_utility


def parse_vnn_lib_prop(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Get all defined variables
    net_inputs, net_outputs, spec_anchors, spec_utility = identify_variables(lines)

    # Input constraints of the form net_inputs >=/<= C [spec_anchors, spec_utility, 1]
    C_lb = np.zeros((len(net_inputs), len(spec_anchors) + len(spec_utility) + 1))
    C_ub = np.zeros((len(net_inputs), len(spec_anchors) + len(spec_utility) + 1))

    # Output constraints of the form C [net_outputs, 1] >= 0
    C_out = np.zeros((0, len(net_outputs) + 1))

    # Dictionaries associating variables with indicies
    idx_dict = {f"{x[0]}_{x[1]}": i for i, x in enumerate(sorted(spec_anchors) + sorted(spec_utility))}
    idx_dict["const_-1"] = -1
    idx_dict_out = {f"{x[0]}_{x[1]}": i for i, x in enumerate(net_outputs)}
    idx_dict_out["const_-1"] = -1

    # Extract all constraints
    for line in lines:
        if line.startswith(";"): continue
        if line.startswith("(assert"):
            match = re.match("\(assert \(([>,<,=]+) (.*)\)\)", line)
            if match is None:
                assert False, line
            else:
                spec_relation = match.group(1)
                spec_content = match.group(2)
                if spec_content.startswith("("):
                    match = re.match("\((.*)\) .*", spec_content)
                    if match is None:
                        assert False, spec_content
                    else:
                        first_term = match.group(1)
                else:
                    match = re.match("([A-Z,a-z,_,\-,\+,0-9,\.]*) .*", spec_content)
                    if match is None:
                        assert False, spec_content
                    else:
                        first_term = match.group(1)
                if spec_content.endswith(")"):
                    match = re.match(".* \((.*)\)", spec_content)
                    if match is None:
                        assert False, spec_content
                    else:
                        second_term = match.group(1)
                else:
                    match = re.match(".* ([A-Z,a-z,_,\.,\-,\+,0-9]*)$", spec_content)
                    if match is None:
                        assert False, spec_content
                    else:
                        second_term = match.group(1)
            assert spec_relation in [">=", "<="]
            g_terms = extract_terms(first_term if spec_relation == ">=" else second_term)
            l_terms = extract_terms(second_term if spec_relation == ">=" else first_term)
            if (len(g_terms) == 1 and g_terms[0][0] == "X"):
                # lower bound on input
                var_idx = g_terms[0][1]
                assert abs(C_lb[var_idx]).sum() == 0, f"multiple lower bounds not supported{g_terms}"
                for term in l_terms:
                    var_key = f"{term[0]}_{term[1]}"
                    assert var_key in idx_dict
                    C_lb[var_idx, idx_dict[var_key]] = term[2]
            elif (len(l_terms) == 1 and l_terms[0][0] == "X"):
                # upper bound on input
                var_idx = l_terms[0][1]
                assert abs(C_ub[var_idx]).sum() == 0, f"multiple upper bounds not supported: {l_terms}"
                for term in g_terms:
                    var_key = f"{term[0]}_{term[1]}"
                    assert var_key in idx_dict
                    C_ub[var_idx, idx_dict[var_key]] = term[2]
            else:
                # output constraint
                C_out_new = np.zeros((1, C_out.shape[1]))
                for term in g_terms:
                    var_key = f"{term[0]}_{term[1]}"
                    assert var_key in idx_dict_out
                    C_out_new[0, idx_dict_out[var_key]] += term[2]
                for term in l_terms:
                    var_key = f"{term[0]}_{term[1]}"
                    assert var_key in idx_dict_out
                    C_out_new[0, idx_dict_out[var_key]] -= term[2]
                C_out = np.concatenate([C_out, C_out_new], axis=0)
    return C_lb, C_ub, C_out


def translate_output_constraints(C_out):
    and_list = []
    for i in range(C_out.shape[0]):
        numeric = C_out[i,-1]
        if numeric != 0:
            l_label = (C_out[i, 0:-1] == -1).nonzero()
            g_label = (C_out[i, 0:-1] == 1).nonzero()
            assert len(l_label) == 1 + len(g_label) == 1
            if len(l_label)>0:
                raise NotImplementedError
            else:
                and_list.append([(l_label, -1, numeric)])
        else:
            l_label = (C_out[i,0:-1]==-1).nonzero()[0]
            g_label = (C_out[i,0:-1]==1).nonzero()[0]
            assert len(l_label)==1 and len(g_label)==1
            and_list.append([(g_label[0], l_label[0], 0)])
    return and_list


def translate_input_to_box(C_lb, C_ub, x_0=None, eps=None, domain_bounds=None):
    n_x = C_lb.shape[0]
    x=[]
    if x_0 is not None:
        if len(x_0) == 1:
            x_0 = np.ones(n_x)*x_0
        else:
            assert len(x_0)==n_x
            x_0 = np.array(x_0)
        n_e = C_lb.shape[1]-1-n_x
        x.append(x_0)
    else:
        n_e = C_lb.shape[1]-1-n_x

    if eps is not None:
        if len(eps) == 1:
            eps = np.ones(n_e)*eps
        else:
            assert len(eps)==n_e
            eps = np.array(eps)
        x.append(eps)

    x.append(np.array([1.]))
    lb = np.matmul(C_lb,np.concatenate(x,axis=0))
    ub = np.matmul(C_ub, np.concatenate(x, axis=0))

    if domain_bounds is not None:
        d_lb, d_ub = domain_bounds
        if len(d_lb) == 1:
            d_lb = np.ones(n_x)*d_lb
        else:
            assert len(d_lb)==n_x
            d_lb = np.array(d_lb)
        if len(d_ub) == 1:
            d_ub = np.ones(n_x)*d_ub
        else:
            assert len(d_ub)==n_x
            d_ub = np.array(d_ub)

        lb = np.maximum(lb, d_lb)
        ub = np.minimum(lb, d_ub)
    return [[(lb[i], ub[i]) for i in range(n_x)]]

def negate_cstr_or_list_old(or_list):
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