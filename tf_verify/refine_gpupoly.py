from optimizer import *
from krelu import *
import time

def refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, true_label, labels_to_be_verified, K=3, s=-2,
                           timeout_lp=10, timeout_milp=10, timeout_final_lp=100, timeout_final_milp=100, use_milp=False,
                           partial_milp=False, max_milp_neurons=30, complete=False, approx=True):
    relu_groups = []
    nlb = []
    nub = []
    #print("INPUT SIZE ", network._lib.getOutputSize(network._nn, 0))
    layerno = 2
    new_relu_layers = []
    for l in range(nn.numlayer):
        num_neurons = network._lib.getOutputSize(network._nn, layerno)
        #print("num neurons ", num_neurons)
        if layerno in relu_layers:
            pre_lbi = nlb[len(nlb)-1]
            pre_ubi = nub[len(nub)-1]
            lbi = np.zeros(num_neurons)
            ubi = np.zeros(num_neurons)
            for j in range(num_neurons):
                lbi[j] = max(0,pre_lbi[j])
                ubi[j] = max(0,pre_ubi[j])
            layerno =  layerno+2
            new_relu_layers.append(len(nlb))
            #print("RELU ")
        else:
            #print("COMING HERE")
            #A = np.zeros((num_neurons,num_neurons), dtype=np.double)
            #print("FINISHED ", num_neurons)
            #for j in range(num_neurons):
            #    A[j][j] = 1
            bounds = network.evalAffineExpr(layer=layerno)
            #print("num neurons", num_neurons)
            lbi = bounds[:,0]
            ubi = bounds[:,1]
            layerno = layerno+1
        nlb.append(lbi)
        nub.append(ubi)

    second_FC = -2
    for i in range(nn.numlayer):
        if nn.layertypes[i] == 'FC':
            if second_FC == -2:
                second_FC = -1
            else:
                second_FC = i
                break

    index = 0 
    for l in relu_layers:
        gpu_layer = l - 1
        layerno = new_relu_layers[index]
        index = index+1

        if config.refine_neurons==True:
            predecessor_index = nn.predecessors[layerno + 1][0] - 1
            if predecessor_index == second_FC:
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
                                                                 use_milp_temp,  candidate_vars, timeout)
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
        #print("total size ", total_size, kact_args)
        A = np.zeros((total_size, num_neurons), dtype=np.double)
        i = 0
        #print("total_size ", total_size)
        for varsid in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                for j in range(len(varsid)):
                    A[i][varsid[j]] = coeffs[j] 
               
                i = i + 1
        bounds=np.zeros(shape=(0, 2))
        max_eqn_per_call = 500
        for i_a in range((int)(np.ceil(A.shape[0] / max_eqn_per_call))):
            A_temp = A[i_a*max_eqn_per_call:(i_a+1)*max_eqn_per_call]
            bounds_temp = network.evalAffineExpr(A_temp, layer=gpu_layer, back_substitute=network.FULL_BACKSUBSTITUTION, dtype=np.double)
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
        start_milp = time.time()
        counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer,
                                                use_milp=True, is_nchw=True, partial_milp=-1, max_milp_neurons=1e6)
        #model.setParam(GRB.Param.TimeLimit, timeout_final_milp) #set later
    else:
        counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer,
                                                use_milp=False, is_nchw=True)
        model.setParam(GRB.Param.TimeLimit, timeout_final_lp)

    model.setParam(GRB.Param.Cutoff, 0.01)

    if partial_milp != 0 and not complete:
        nn.ffn_counter = 0
        nn.conv_counter = 0
        nn.pool_counter = 0
        nn.concat_counter = 0
        nn.tile_counter = 0
        nn.residual_counter = 0
        nn.activation_counter = 0
        counter_partial_milp, var_list_partial_milp, model_partial_milp = create_model(nn, nn.specLB, nn.specUB, nlb,
                                                                                       nub, relu_groups, nn.numlayer,
                                                                                       complete, is_nchw=True,
                                                                                       partial_milp=partial_milp,
                                                                                       max_milp_neurons=max_milp_neurons)
        model_partial_milp.setParam(GRB.Param.TimeLimit, timeout_final_milp)
        model_partial_milp.setParam(GRB.Param.Cutoff, 0.01)

        # a1=time.time()
        # model_partial_milp.optimize()
        # if model_partial_milp.Status==3:
        #     model_partial_milp.setParam(GRB.Param.FeasibilityTol, 1e-4)
        #     print(f"Infeasible model encountered. Increased tolerance")
        # a2=time.time()
        # print(f"time for feasibility check: {a2-a1:.3f}")

    # num_var = len(var_list)
    #output_size = num_var - counter
    #print("TIMEOUT ", config.timeout_lp)
    flag = True
    x = None
    for label in labels_to_be_verified:
        obj = LinExpr()
        #obj += 1*var_list[785]
        obj += 1*var_list[counter + true_label]
        obj += -1*var_list[counter + label]
        model.setObjective(obj,GRB.MINIMIZE)
        # model.optimize()
        if complete:
            milp_timeout = config.timeout_final_milp if config.timeout_complete is None else (config.timeout_complete + start_milp - time.time())
            model.setParam(GRB.Param.TimeLimit, milp_timeout)
            model.optimize(milp_callback)
        else:
            model.optimize(lp_callback)
        # model.computeIIS()
        #model.write("model_refinegpupo.ilp")
        try:
            print(
                f"Model status: {model.Status}, Obj val/bound against label {label}: {model.objval:.4f}/{model.objbound:.4f}, Final solve time: {model.Runtime:.3f}")
        except:
            print(
                f"Model status: {model.Status}, Objval retrival failed, Final solve time: {model.Runtime:.3f}")

        if model.Status == 6 or (model.Status in [2,11] and model.objval > 0):
            # Cutoff active, or optimal with positive objective => sound against adv_label
            pass
        elif partial_milp != 0 and not complete:
            if model.Status == 2:
                x_adv = np.array(model.x[0:len(nn.specLB)])
                is_not_shown_unsafe = network.test(x_adv, x_adv, int(true_label))
            else:
                is_not_shown_unsafe = True
            if not is_not_shown_unsafe:
                flag = False
                print(f"Counterexample found, partial MILP skipped.")
            else:
                obj = LinExpr()
                obj += 1 * var_list_partial_milp[counter_partial_milp + true_label]
                obj += -1 * var_list_partial_milp[counter_partial_milp + label]
                model_partial_milp.setObjective(obj, GRB.MINIMIZE)
                model_partial_milp.optimize(milp_callback)
                if model_partial_milp.Status == 3:
                    model_partial_milp.setParam(GRB.Param.FeasibilityTol, 1e-4)
                    print(f"Infeasible model encountered. Increased tolerance")
                    model_partial_milp.reset()
                    model_partial_milp.optimize(milp_callback)
                try:
                    print(
                        f"Partial MILP model status: {model_partial_milp.Status}, Obj val/bound against label {label}: {model_partial_milp.ObjVal:.4f}/{model_partial_milp.ObjBound:.4f}, Final solve time: {model_partial_milp.Runtime:.3f}")
                except:
                    print(
                        f"Partial MILP model status: {model_partial_milp.Status}, Obj bound retrival failed, Final solve time: {model_partial_milp.Runtime:.3f}")

                if model_partial_milp.Status in [2, 9, 6, 11] and model_partial_milp.ObjBound > 0:
                    pass
                elif model_partial_milp.Status not in [2, 9, 6, 11]:
                    print("Partial milp model was not successful status is", model_partial_milp.Status)
                    model_partial_milp.write("final.mps")
                    flag = False
                else:
                    flag = False
        elif model.Status not in [2, 11, 6]:
            print("Model was not successful status is",
                  model.Status)
            model.write("final.mps")
            flag = False
        else:
            flag = False
        if not flag and model.Status == 2 and model.objval < 0:
            if model.objval != math.inf:
                x = model.x[0:len(nn.specLB)]

        if not flag:
                break
    return flag, x