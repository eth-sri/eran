from gurobipy import *
import numpy as np
import time

def handle_conv(model,var_list,start_counter, filters,biases,filter_size, numfilters,input_shape, strides, padding, lbi, ubi, use_milp):
    if(padding==True):
       o1 = int(np.ceil((input_shape[0] - filter_size[0]+1)/strides[0]))
       o2 = int(np.ceil((input_shape[1] - filter_size[1]+1)/strides[1]))
    else:
       o1 = int(np.ceil(input_shape[0] / strides[0]))
       o2 = int(np.ceil(input_shape[1] / strides[1]))

    o3 = int(numfilters)
    num_out_neurons = o1*o2*o3
    num_in_neurons = input_shape[0]*input_shape[1]*input_shape[2]

    pad_along_height=0
    pad_along_width=0
    pad_top=0
    pad_left=0
    tmp=0
    
    if(padding==False):
        if (input_shape[0] % strides[0] == 0):
            tmp = filter_size[0] - strides[0]
            pad_along_height = np.max(tmp, 0)
        else:
            tmp = filter_size[0] - (input_shape[0] % strides[0])
            pad_along_height = np.max(tmp, 0)
		
        if (input_shape[1] % strides[1] == 0):
            tmp = filter_size[1] - strides[1]
            pad_along_width = np.max(tmp, 0)
		
        else:
            tmp = filter_size[1] - (input_shape[1] % strides[1]);
            pad_along_width = np.max(tmp, 0);
		
        pad_top = int(np.ceil(pad_along_height / 2))
		
        pad_left = int(np.ceil(pad_along_width / 2))
		

    start = len(var_list)
    for j in range(num_out_neurons):
        var_name = "x" + str(start+j)   
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub =ubi[j], name=var_name)
        var_list.append(var)

    for out_x in range(o1):
        for out_y in range(o2):
            for out_z in range(o3):
                dst_ind = out_x*o2*o3 + out_y*o3 + out_z;
                expr = LinExpr()
                expr += -1*var_list[start+dst_ind]
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
                             filter_index = x_shift*filter_size[1]*input_shape[2]*o3 + y_shift*input_shape[2]*o3 + inp_z*o3 + out_z
                             #expr.addTerms(filters[filter_index],var_list[src_ind])
                             expr.addTerms(filters[x_shift][y_shift][inp_z][out_z],var_list[src_ind])	
                
                expr.addConstant(biases[out_z])
                model.addConstr(expr, GRB.EQUAL, 0)	
    return start
			     

def handle_maxpool(model,var_list,layerno,src_counter, pool_size, input_shape, lbi,ubi,lbi_prev, ubi_prev,use_milp):
   
    start = len(var_list)
    num_neurons = input_shape[0]*input_shape[1]*input_shape[2]
    binary_counter = start
    maxpool_counter = start
    if(use_milp==1):
        maxpool_counter = start + num_neurons
        for j in range(num_neurons):
            var_name = "x" + str(start+j)
            var = model.addVar(vtype=GRB.BINARY, name=var_name)

            var_list.append(var)
    o1 = int(input_shape[0]/pool_size[0])
    o2 = int(input_shape[1]/pool_size[1])
    o3 = int(input_shape[2]/pool_size[2])
    output_size = o1*o2*o3

    for j in range(output_size):
        var_name = "x" + str(maxpool_counter+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = lbi[j], ub=ubi[j],  name=var_name)
        var_list.append(var) 
    
    output_offset = 0
    for out_x in range(o1):
        for out_y in range(o2):
            for out_z in range(o3):
                sum_u = 0.0
                sum_l = 0.0
                max_u = float("-inf")
                max_l = float("-inf")
                pool_map = []
                inf = []
                sup = []
                l = 0
                for x_shift in range(pool_size[0]):
                    for y_shift in range(pool_size[1]):
                        x_val = out_x*2 + x_shift
                        y_val = out_y*2 + y_shift  
                        mat_offset = x_val*input_shape[1]*input_shape[2] + y_val*input_shape[2] + out_z
                        pool_cur_dim = src_counter + mat_offset
                        pool_map.append(mat_offset)
                        inf.append(lbi_prev[mat_offset])				
                        sup.append(ubi_prev[mat_offset])
                        sum_u = sum_u + sup[l];
                        sum_l = sum_l + inf[l];
                        if(sup[l]>max_u):
                           max_u = sup[l]
                        if(inf[l] > max_l):
                           max_l = inf[l]
                        l = l+1
                dst_index = maxpool_counter+output_offset		
                p01 = pool_size[0]*pool_size[1]
                if(use_milp==1):
                    binary_expr = LinExpr()
                    for l in range(p01):
                       src_index = pool_map[l]
                       src_var = src_index + src_counter
                       binary_var = src_index + binary_counter
                       if(ubi_prev[src_index]<max_l):
                           continue
 
                       # y >= x
                      
                       expr = var_list[dst_index] -  var_list[src_var]
                       model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                       # y <= x + (1-a)*(u_{rest}-l)
                       max_u_rest = float("-inf")
                       for j in range(p01):
                           if(j==l):
                              continue
                           if(sup[j]>max_u_rest):
                              max_u_rest = sup[j]
	               
                       cst = max_u_rest-inf[l]
                       
                       expr = var_list[dst_index] - var_list[src_var] + cst*var_list[binary_var]  
                       model.addConstr(expr, GRB.LESS_EQUAL, cst)
                 
	               # indicator constraints
                       model.addGenConstrIndicator(var_list[binary_var], True, var_list[dst_index]-var_list[src_var], GRB.EQUAL, 0.0)
                        
                       binary_expr+=var_list[binary_var]

                    model.addConstr(binary_expr, GRB.EQUAL, 1)

                else:
                    add_expr = LinExpr()
                    add_expr+=-1*var_list[dst_index]
                    for l in range(p01):
                        src_index = pool_map[l]
                        src_var = src_index + src_counter
                        # y >= x
                        expr = var_list[dst_index] - var_list[src_var]
                        model.addConstr(expr, GRB.GREATER_EQUAL, 0)
                        
                        add_expr+=var_list[src_var]
                    model.addConstr(add_expr, GRB.GREATER_EQUAL, sum_l - max_l)
                    
                output_offset += 1 
 
    return maxpool_counter


def handle_affine(model,var_list,counter,weights,biases, lbi, ubi):
    num_neurons_affine = len(weights)
    start = len(var_list)
    
    # output of matmult
    for j in range(num_neurons_affine):
        var_name = "x" + str(start+j)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=lbi[j], ub =ubi[j], name=var_name)
        var_list.append(var)
    
    for j in range(num_neurons_affine):
        num_in_neurons = len(weights[j])
        
        expr = LinExpr()
        expr += -1*var_list[start+j]
        # matmult constraints
        for k in range(num_in_neurons):
            expr.addTerms(weights[j][k],var_list[counter+k])
        expr.addConstant(biases[j])
        model.addConstr(expr, GRB.EQUAL, 0)
    return start
    


def handle_relu(model,var_list,layerno,affine_counter,num_neurons,lbi,ubi,use_milp):
    start= len(var_list)
    binary_counter = start
    relu_counter = start
    
    if(use_milp==1):
        #indicator variables
        relu_counter = start + num_neurons
        for j in range(num_neurons):
           var_name = "x" + str(start+j)
           var = model.addVar(vtype=GRB.BINARY, name=var_name)
           var_list.append(var)
   
    # relu variables
    for j in range(num_neurons):
        var_name = "x" + str(relu_counter+j)
        upper_bound = max(0,ubi[j])
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = 0.0, ub=upper_bound,  name=var_name)
        var_list.append(var)    

    
    for j in range(num_neurons):
        if(ubi[j]<=0):
           expr = var_list[relu_counter+j]
           model.addConstr(expr, GRB.EQUAL, 0)
        elif(lbi[j]>=0):
           expr = var_list[relu_counter+j] - var_list[affine_counter+j]
           model.addConstr(expr, GRB.EQUAL, 0)
        else:
           if(use_milp==1):
               
	       # y <= x - l(1-a)
               expr = var_list[relu_counter+j] - var_list[affine_counter+j] - lbi[j]*var_list[binary_counter+j]  
               model.addConstr(expr, GRB.LESS_EQUAL, -lbi[j])
                 
               # y >= x
               expr = var_list[relu_counter+j] -  var_list[affine_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)

               # y <= u.a
               expr = var_list[relu_counter+j] - ubi[j]*var_list[binary_counter+j]
               model.addConstr(expr, GRB.LESS_EQUAL, 0)

               # y >= 0
               expr = var_list[relu_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)
		 
	       # indicator constraint
               model.addGenConstrIndicator(var_list[binary_counter+j], True, var_list[affine_counter+j], GRB.GREATER_EQUAL, 0.0)
           else:
               # y >= 0
               
               expr = var_list[relu_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)
  
               # y >= x
               expr = var_list[relu_counter+j] - var_list[affine_counter+j]
               model.addConstr(expr, GRB.GREATER_EQUAL, 0)

               # y <= lambda.x + mu
               slope = ubi[j]/(ubi[j]-lbi[j])
               intercept = -slope*lbi[j]
               expr = var_list[relu_counter+j] - slope*var_list[affine_counter+j]
               model.addConstr(expr, GRB.LESS_EQUAL, intercept)

    return relu_counter

def create_model(nn, LB_N0, UB_N0, nlb, nub, numlayer, use_milp, relu_needed):
    

    model = Model("milp")
   
    model.setParam("OutputFlag",0)
    num_pixels = len(LB_N0)
    #output_counter = num_pixels
    ffn_counter = nn.ffn_counter 
    conv_counter = nn.conv_counter
    maxpool_counter = nn.maxpool_counter
    
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0 
    var_list = []
    
    for i in range(num_pixels):
        var_name = "x" + str(i)
        var = model.addVar(vtype=GRB.CONTINUOUS, lb = LB_N0[i], ub=UB_N0[i], name=var_name)
        var_list.append(var)

    counter = 0
    start = 0
    for i in range(numlayer):
        if(nn.layertypes[i]=='SkipNet1'):
            start = i+1
            #break
        elif(nn.layertypes[i]=='SkipNet2'):
            start = i+1
            #break

    for i in range(start):
        if(nn.layertypes[i] in ['ReLU','Affine']):
            nn.ffn_counter+=1          
        elif(nn.layertypes[i]=='Conv2D'):
            nn.conv_counter+=1
        elif(nn.layertypes[i]=='MaxPooling2D'):
            nn.maxpool_counter+=1
       
     
    for i in range(start,numlayer):
        if(nn.layertypes[i] in ['SkipCat']):
            continue 
        elif(nn.layertypes[i] in ['ReLU','Affine']):
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter+nn.conv_counter]
            counter = handle_affine(model,var_list,counter,weights,biases,nlb[i-start],nub[i-start])
            
            if(nn.layertypes[i]=='ReLU' and relu_needed[i]):
                counter = handle_relu(model,var_list,i,counter,len(weights),nlb[i-start],nub[i-start],use_milp)
            
            nn.ffn_counter+=1


        elif(nn.layertypes[i]=='Conv2D'):
            filters = nn.filters[nn.conv_counter]
            biases = nn.biases[nn.ffn_counter+nn.conv_counter]
            filter_size = nn.filter_size[nn.conv_counter]
            numfilters = nn.numfilters[nn.conv_counter]
            strides = nn.strides[nn.conv_counter]
            padding = nn.padding[nn.conv_counter]
            input_shape = nn.input_shape[nn.conv_counter +nn.maxpool_counter]
            if(padding==True):
               o1 = int(np.ceil((input_shape[0] - filter_size[0]+1)/strides[0]))
               o2 = int(np.ceil((input_shape[1] - filter_size[1]+1)/strides[1]))
            else:
               o1 = int(np.ceil(input_shape[0] / strides[0]))
               o2 = int(np.ceil(input_shape[1] / strides[1]))
            o3 = numfilters
            num_neurons = o1*o2*o3
            counter = handle_conv(model,var_list, counter, filters,biases,filter_size,numfilters,input_shape,strides, padding, nlb[i-start],nub[i-start],use_milp)
            if(relu_needed[i]):
               counter = handle_relu(model,var_list,i,counter,num_neurons,nlb[i-start],nub[i-start],use_milp)
            nn.conv_counter+=1


        elif(nn.layertypes[i]=='MaxPooling2D'):
            pool_size = nn.pool_size[nn.maxpool_counter]
            input_shape = nn.input_shape[nn.conv_counter + nn.maxpool_counter]
            maxpool_lb = nn.maxpool_lb[nn.maxpool_counter]
            maxpool_ub = nn.maxpool_ub[nn.maxpool_counter]
            counter = handle_maxpool(model,var_list,i,counter,pool_size, input_shape, nlb[i-start],nub[i-start],maxpool_lb,maxpool_ub,use_milp)
            nn.maxpool_counter+=1


        else:
            print('layertype not supported')
            return
    nn.ffn_counter = ffn_counter
    nn.conv_counter = conv_counter
    nn.maxpool_counter = maxpool_counter
    return counter, var_list, model


def get_bounds_for_layer_with_milp(nn, LB_N0, UB_N0, layerno, abs_layer_count, output_size, nlb, nub, use_milp, candidate_vars, timeout):
    
    is_conv = False

    for i in range(nn.numlayer):
        if nn.layertypes[i] == 'Conv2D':
            is_conv = True
            break
    relu_needed = [0]*(layerno+1)
    for i in range(layerno):
        relu_needed[i] = 1

    
    lbi = nlb[abs_layer_count]
    ubi = nub[abs_layer_count]
    numlayer = nn.numlayer
    keep_bounds = [1]*len(lbi)
    candidate_length = len(candidate_vars)
    widths = np.zeros(candidate_length)
    avg_weight = np.zeros(candidate_length)
    next_layer = nn.ffn_counter +  nn.conv_counter + nn.maxpool_counter + 1
  
    for i in range(candidate_length):
        ind = candidate_vars[i]
        keep_bounds[ind] = 0
        widths[i] = ubi[ind]-lbi[ind]

        if next_layer < numlayer:
            #weights = nn.weights[next_layer]
            if(nn.layertypes[layerno]in ['ReLU','Affine']):
                weights = nn.weights[nn.ffn_counter+1]
                for j in range(len(weights)):
                    avg_weight[i]+=abs(weights[j][ind])
    sorted_width_indices = np.argsort(widths)
    sorted_avg_weight_indices = np.argsort(avg_weight)
   

    features = []

    for i in range(candidate_length):
        features.append(sorted_width_indices[i] + sorted_avg_weight_indices[i])
    sorted_features_indices = sorted(range(len(features)), key=lambda k: features[k],reverse=True)

   
    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, layerno+1, use_milp, relu_needed)
    
    
    
    resl = [0]*len(lbi)
    resu = [0]*len(lbi)
    indices = []
   
    num_candidates = 0
    if is_conv==True:
       
        num_candidates = int(len(candidate_vars))
        
    else:
        if(abs_layer_count<=3):
            num_candidates = int(len(candidate_vars)/math.pow(5,abs_layer_count-1))
            #num_candidates = len(candidate_vars)
        else:
            num_candidates = int(len(candidate_vars)/math.pow(2,abs_layer_count-4))
    
    neuron_map = [0]*len(lbi)
  
    solvetime = 0
    
    model.setParam('TimeLimit', timeout)
    output_counter = counter

    for i in range(num_candidates):
        ind = candidate_vars[sorted_features_indices[i]]
        neuron_map[ind] =1
        
        #if(lbi[ind]>ubi[ind]):
        obj = LinExpr()
        obj += var_list[output_counter+ind]
        model.setObjective(obj,GRB.MINIMIZE)
        model.optimize()
        solvetime += model.RunTime
        result_l = None
        result_u = None
        flag1 = True
        flag2 = True
        if(model.SolCount==0):
            flag1 = False
        else:    
            result_l = model.objbound
        model.setObjective(obj,GRB.MAXIMIZE)
        model.optimize()
        
        solvetime += model.RunTime
        if(model.SolCount==0):
            flag2 = False
        else:
            result_u = model.objbound

        if(flag1==True):
            if(flag2==True):
                if(result_u > result_l):
                    resl[ind] = max(result_l,lbi[ind])
                    resu[ind] = min(result_u,ubi[ind])
                    indices.append(ind)
                else:
                    resl[ind] = lbi[ind]
                    resu[ind] = ubi[ind]
            else:
                resl[ind] = max(lbi[ind],result_l)
                resu[ind] = ubi[ind]
                indices.append(ind)
        else:
            resl[ind] = lbi[ind]
            if((flag2==True) and (result_u > lbi[ind])):
                resu[ind] = min(result_u,ubi[ind])
                indices.append(ind)
            else:
                resl[ind] = lbi[ind]
                resu[ind] = ubi[ind] 
            
            
                
      
    avg_solvetime = (solvetime+1)/(2*num_candidates+1)
    
    model.setParam('TimeLimit', avg_solvetime/2)
    for j in range(output_size):
        if(not neuron_map[j]):
           if(keep_bounds[j]):
               
               resl[j] = lbi[j]
               resu[j] = ubi[j]
           else:
               flag1 = True
               flag2 = True
               result_l = None
               result_u = None
               obj = LinExpr()
               obj += var_list[output_counter+j]
               model.setObjective(obj,GRB.MINIMIZE)
               model.optimize()
               
               if(model.SolCount==0):
                   flag1 = False
               else:
                   result_l = model.objbound
                   
               model.setObjective(obj,GRB.MAXIMIZE)
               model.optimize()
               if(model.SolCount==0):
                   flag2 = False
               else:
                   result_u = model.objbound
               if(flag1==True):
                   if(flag2==True):
                       if(result_u > result_l):
                           resl[j] = max(result_l,lbi[j])
                           resu[j] = min(result_u,ubi[j])
                           indices.append(j)
                       else:
                           resl[j] = lbi[j]
                           resu[j] = ubi[j]
                   else:
                      resl[j] = max(lbi[j],result_l)
                      resu[j] = ubi[j]
                      indices.append(j)
               else:
                   resl[j] = lbi[j]
                   if(flag2==True):
                       resu[j] = min(result_u,ubi[j])
                       indices.append(j)
                   else:
                       resl[j] = lbi[j]
                       resu[j] = ubi[j]                    
               
    
    for i in range(abs_layer_count):
        for j in range(len(nlb[i])):
            if(nlb[i][j]>nub[i][j]):
                print("fp unsoundness detected ", nlb[i][j],nub[i][j],i,j)


    return resl, resu, sorted(indices)

def verify_network_with_milp(nn, LB_N0, UB_N0, c, nlb, nub, is_max=True):
    nn.ffn_counter = 0
    nn.conv_counter = 0
    nn.maxpool_counter = 0
    use_milp = []
    relu_needed = []
    input_size = len(LB_N0)
    numlayer = nn.numlayer
    for i in range(numlayer):
        use_milp.append(1)
        relu_needed.append(1)
    
    counter, var_list, model = create_model(nn, LB_N0, UB_N0, nlb, nub, numlayer, True,relu_needed)
    
    num_var = len(var_list)
    output_size = num_var - counter
   
    for i in range(output_size):
        if(i!=c):
            obj = LinExpr()
            if is_max:
                obj += 1*var_list[counter+c]
                obj += -1*var_list[counter + i]
            else:
                obj += -1*var_list[counter+c]
                obj += 1*var_list[counter + i]
            model.setObjective(obj,GRB.MINIMIZE)
            model.optimize()
            
            if(model.objval<0):  
                        
                return False, model.x[0:input_size]
   
    return True, model.x[0:input_size]
  
