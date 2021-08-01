from optimizer import *
from krelu import *
from constraint_utils import get_constraints_for_dominant_label
import time
from ai_milp import evaluate_models


def refine_gpupoly_results(nn, network, config, relu_layers, true_label, adv_labels=-1, K=3, s=-2,
                           timeout_lp=10, timeout_milp=10, timeout_final_lp=100, timeout_final_milp=100, use_milp=False,
                           partial_milp=False, max_milp_neurons=30, complete=False, approx=True, constraints=None,
                           terminate_on_failure=True, max_eqn_per_call=500, eval_instance=None, find_adex=False,
                           start_time=None, feas_act=None):
    # nn.predecessors = []
    # for pred in range(0, nn.numlayer + 1):
    #     predecessor = np.zeros(1, dtype=np.int)
    #     predecessor[0] = int(pred - 1)
    #     nn.predecessors.append(predecessor)
    # print("predecessors ", nn.predecessors[0][0])

    relu_groups = []
    nlb = []
    nub = []
    if constraints is None:
        constraints = []

    #print("INPUT SIZE ", network._lib.getOutputSize(network._nn, 0))
    layerno = 0
    new_relu_layers = []
    for l in range(nn.numlayer):
        # num_neurons = network._lib.getOutputSize(network._nn, layerno)
        if nn.layertypes[l] in ["FC", "Conv",]:
            layerno += 2
        else:
            layerno += 1
        if layerno in relu_layers:
            pre_lbi = nlb[len(nlb)-1]
            pre_ubi = nub[len(nub)-1]
            lbi = np.maximum(0, pre_lbi)
            ubi = np.maximum(0, pre_ubi)
            new_relu_layers.append(len(nlb))
            #print("RELU ")
        else:
            bounds = network.evalAffineExpr(layer=layerno, sound=config.fp_sound)
            lbi = bounds[:, 0]
            ubi = bounds[:, 1]
        nlb.append(lbi)
        nub.append(ubi)

    second_FC = -2
    for i in range(nn.numlayer):
        if nn.layertypes[i] in ['FC', "Conv"]:
            if second_FC == -2:
                second_FC = -1
            else:
                second_FC = i
                break
    affine_layers = np.array([x=="Conv" or x=="FC" for x in nn.layertypes]).nonzero()[0]

    index = 0
    for l in relu_layers:
        layerno = new_relu_layers[index]
        index += 1

        if config.refine_neurons==True:
            predecessor_index = nn.predecessors[layerno + 1][0] - 1
            if 0 < sum((affine_layers >= second_FC).__and__(predecessor_index >= affine_layers)) <= config.n_milp_refine:  # predecessor_index >= second_FC :#and domain=="deepzono" and predecessor_index>second_FC:
                use_milp_temp = use_milp
                timeout = timeout_milp
            else:
                use_milp_temp = False
                timeout = timeout_lp
            length = len(nlb[predecessor_index])

            candidate_vars = []
            for i in range(length):
                if ((nlb[predecessor_index][i] < 0 and nub[predecessor_index][i] > 0) or (nlb[predecessor_index][i] > 0)):
                    candidate_vars.append(i)

            start = time.time()
            resl, resu, indices = get_bounds_for_layer_with_milp(nn, nn.specLB, nn.specUB, predecessor_index,
                                                                 predecessor_index, length, nlb, nub, relu_groups,
                                                                 use_milp_temp,  candidate_vars, timeout,
                                                                 gpu_model=True, start_time=start_time, is_nchw=True, feas_act=feas_act)
            end = time.time()
            if config.debug:
                print(f"Refinement of bounds time: {end-start:.3f}. MILP used: {use_milp_temp}")
            nlb[predecessor_index] = resl
            nub[predecessor_index] = resu

        lbi = nlb[layerno-1]
        ubi = nub[layerno-1]
        #print("LBI ", lbi, "UBI ", ubi, "specLB")
        num_neurons = len(lbi)

        kact_args = sparse_heuristic_with_cutoff(num_neurons, lbi, ubi, K=K, s=s)
        kact_cons = []
        total_size = 0
        for varsid in kact_args:
            size = 3**len(varsid) - 1
            total_size = total_size + size
        if config.debug:
            print(f"Number of input constraints: {total_size}, number of groups {len(kact_args)}")
        A = np.zeros((total_size, num_neurons), dtype=config.dtype)
        i = 0
        for varsid in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                for j in range(len(varsid)):
                    A[i][varsid[j]] = coeffs[j]

                i = i + 1
        bounds = np.zeros(shape=(0, 2))
        for i_a in range((int)(np.ceil(A.shape[0] / max_eqn_per_call))):
            A_temp = A[i_a*max_eqn_per_call:(i_a+1)*max_eqn_per_call]
            bounds_temp = network.evalAffineExpr(A_temp, layer=l-1, back_substitute=network.FULL_BACKSUBSTITUTION, sound=config.fp_sound)#, dtype=np.double)
            bounds = np.concatenate([bounds, bounds_temp], axis=0)
        upper_bound = bounds[:,1]
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
        KAct.type = "ReLU"
        with multiprocessing.Pool(config.numproc) as pool:
            # kact_results = pool.map(make_kactivation_obj, input_hrep_array)
            kact_results = list(pool.starmap(make_kactivation_obj, zip(input_hrep_array, len(input_hrep_array) * [approx])))

        gid = 0
        for inst in kact_results:
            varsid = kact_args[gid]
            inst.varsid = varsid
            kact_cons.append(inst)
            gid = gid+1
        relu_groups.append(kact_cons)

    if complete:
        counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer,
                                                use_milp=True, is_nchw=True, partial_milp=-1, max_milp_neurons=-1,
                                                gpu_model=True, feas_act=feas_act)
    else:
        counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer,
                                                use_milp=False, is_nchw=True, gpu_model=True)
        model.setParam(GRB.Param.TimeLimit, timeout_final_lp)

    if not (config.regression and config.epsilon_y == 0):
        model.setParam(GRB.Param.Cutoff, 0.01)

    if partial_milp != 0:
        counter_partial_milp, var_list_partial_milp, model_partial_milp = create_model(nn, nn.specLB, nn.specUB, nlb,
                                                                                       nub, relu_groups, nn.numlayer,
                                                                                       False, is_nchw=True,
                                                                                       partial_milp=partial_milp,
                                                                                       max_milp_neurons=max_milp_neurons,
                                                                                       feas_act=feas_act)
        model_partial_milp.setParam(GRB.Param.TimeLimit, timeout_final_milp)

        if not (config.regression and config.epsilon_y == 0):
            model_partial_milp.setParam(GRB.Param.Cutoff, 0.01)
    else:
        model_partial_milp = None
        var_list_partial_milp = None
        counter_partial_milp = None

    # num_var = len(var_list)
    #output_size = num_var - counter
    #print("TIMEOUT ", config.timeout_lp)
    flag = True
    x = None

    if len(constraints) == 0 or adv_labels != -1:
        if config.regression:
            constraints.append([(0, -1, true_label - config.epsilon_y)])
            constraints.append([(-1, 0, true_label + config.epsilon_y)])
        else:
            num_outputs = len(nn.weights[-1])
            # Matrix that computes the difference with the expected layer.
            diffMatrix = np.delete(-np.eye(num_outputs), true_label, 0)
            diffMatrix[:, true_label] = 1
            diffMatrix = diffMatrix.astype(np.float64)

            # gets the values from GPUPoly.
            res = network.evalAffineExpr(diffMatrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO, sound=config.fp_sound)

            var = 0
            for label in range(num_outputs):
                if label != true_label:
                    if res[var][0] < 0:
                        # add constraints that could not be proven using standard gpupoly for evaluation with
                        constraints.append([(true_label, label, 0)])
                    var = var + 1

    constraints_hold, failed_constraints, adex_list, model_bounds = evaluate_models(model, var_list, counter,
                                                                                    len(nn.specLB), constraints,
                                                                                    terminate_on_failure,
                                                                                    model_partial_milp,
                                                                                    var_list_partial_milp,
                                                                                    counter_partial_milp, eval_instance,
                                                                                    find_adex=find_adex, start_time=start_time,
                                                                                    dtype=config.dtype)


    dominant_class = true_label if constraints_hold else -1

    failed_constraints = failed_constraints if len(failed_constraints) > 0 else None
    adex_list = adex_list if len(adex_list) > 0 else None

    return constraints_hold, nn, nlb, nub, failed_constraints, adex_list, model_bounds
