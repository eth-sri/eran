from optimizer import *
from krelu import *

def refine_gpupoly_results(nn, specLB, specUB, network, num_gpu_layers, relu_layers):
    relu_groups = []
    for l in relu_layers:
        layerno = l - 1
        
        num_neurons = network._lib.getOutputSize(network._nn, layerno)
        A = np.zeros((num_neurons,num_neurons), dtype=np.double)
        
        for j in range(num_neurons):
            A[j][j] = 1
    
        bounds = network.evalAffineExpr(A, layer=layerno, back_substitute=network.FULL_BACKSUBSTITUTION, dtype=np.double)
        lbi = bounds[:,0]
        ubi = bounds[:,1]
        print("upper bound", ubi<=0)
        kact_args = sparse_heuristic_with_cutoff(num_neurons, lbi, ubi)
        kact_cons = []
        total_size = 0
        for varsid in kact_args:
            size = 3**len(varsid) - 1
            total_size = total_size + size
        A = np.zeros((total_size, num_neurons), dtype=np.double)
        i = 0
        for varsid in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                for j in range(len(varsid)):
                    A[i][varsid[j]] = coeffs[j] 
               
                i = i + 1
        bounds = network.evalAffineExpr(A, layer=layerno, back_substitute=network.FULL_BACKSUBSTITUTION, dtype=np.double)
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
            kact_results = pool.map(make_kactivation_obj, input_hrep_array)
        gid = 0
        for inst in kact_results:
            varsid = kact_args[gid]
            inst.varsid = varsid
            kact_cons.append(inst)
            gid = gid+1
    relu_groups.append(kact_cons)
    #counter, var_list, model = create_model(nn, specLB, specUB, nlb, nub, relu_groups, nn.numlayer, config.complete==True)
    #model.setParam(GRB.Param.TimeLimit,self.timeout_lp)
    #num_var = len(var_list)
    #output_size = num_var - counter
    #for i in output_size:
    #    for j in output_size:
    #        if i!=j:
    #            obj += 1*var_list[counter + i]
    #            obj += -1*var_list[counter + j]
    #            model.setObjective(obj,GRB.MINIMIZE)
    #            model.optimize()
    #            if model.Status!=2:
    #                print("model was not successful status is", model.Status)
    #                model.write("final.mps")
    #                flag = False
    #                break                       
    #            if model.objbound > 0:
    #               or_result = True
    #                    #print("objbound ", model.objbound)
    #               if model.solcount > 0:
    #                    non_adv_examples.append(model.x[0:input_size])
    #                    break
    #                elif model.solcount > 0:
    #                    adv_examples.append(model.x[0:input_size])
                
