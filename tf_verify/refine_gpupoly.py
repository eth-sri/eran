from optimizer import *
from krelu import *
import time

def refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, true_label, labels_to_be_verified, K=3, timeout_lp=10, timeout_milp=10, timeout_final_lp=100, use_milp=False):
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

        kact_args = sparse_heuristic_with_cutoff(num_neurons, lbi, ubi, K=K)
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
        # with multiprocessing.Pool(config.numproc) as pool:
        #     kact_results = pool.map(make_kactivation_obj, input_hrep_array)
        kact_results = list(map(make_kactivation_obj, input_hrep_array))

        gid = 0
        for inst in kact_results:
            varsid = kact_args[gid]
            inst.varsid = varsid
            kact_cons.append(inst)
            gid = gid+1
        relu_groups.append(kact_cons)
    counter, var_list, model = create_model(nn, nn.specLB, nn.specUB, nlb, nub, relu_groups, nn.numlayer, config.complete==True, is_nchw=True)
    model.setParam(GRB.Param.TimeLimit, timeout_final_lp)
    model.setParam(GRB.Param.Cutoff, 0.01)

    num_var = len(var_list)
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
        model.optimize()
        # model.optimize(lp_callback)
        # model.computeIIS()
        #model.write("model_refinegpupo.ilp")
        try:
            print(f"Model status: {model.Status}, Objval against label {j}: {model.objval}, Final solve time: {model.Runtime}")
        except:
            print(f"Model status: {model.Status}, Objval retrival failed, Final solve time: {model.Runtime}")
        if model.Status == 6:
            pass
        elif model.Status!=2:
            print("model was not successful status is", model.Status)
            model.write("final.mps")
            flag = False
            break                       
        elif model.objval < 0:
                               
            flag = False
            if model.objval != math.inf:
                  x = model.x[0:len(nn.specLB)]
            break
    return flag, x