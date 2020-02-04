import tensorflow as tf
import numpy as np
import re
import onnx

def product(it):
    product = 1
    for x in it:
        product *= x
    return product

def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")

def extract_mean(text):
    mean = ''
    m = re.search('mean=\[(.+?)\]', text)
    
    if m:
        means = m.group(1)
    mean_str = means.split(',')
    num_means = len(mean_str)
    mean_array = np.zeros(num_means)
    for i in range(num_means):
         mean_array[i] = np.float64(mean_str[i])
    return mean_array

def extract_std(text):
    std = ''
    m = re.search('std=\[(.+?)\]', text)
    if m:
        stds = m.group(1)
    std_str =stds.split(',')
    num_std = len(std_str)
    std_array = np.zeros(num_std)
    for i in range(num_std):
        std_array[i] = np.float64(std_str[i])
    return std_array

def numel(x):
    return product([int(i) for i in x.shape])

def parseVec(net):
    return np.array(eval(net.readline()[:-1]))

def myConst(vec):
    return tf.constant(vec.tolist(), dtype = tf.float64)

def permutation(W, h, w, c):
    m = np.zeros((h*w*c, h*w*c))
    
    column = 0
    for i in range(h*w):
        for j in range(c):
            m[i+j*h*w, column] = 1
            column += 1
    
    return np.matmul(W, m) 

tf.InteractiveSession().as_default()



def read_tensorflow_net(net_file, in_len, is_trained_with_pytorch):
    mean = 0.0
    std = 0.0
    net = open(net_file,'r')
    x = tf.placeholder(tf.float64, [in_len], name = "x")
    y = None
    z1 = None
    z2 = None
    last_layer = None
    h,w,c = None, None, None
    is_conv = False
    while True:
        curr_line = net.readline()[:-1]
        if 'Normalize' in curr_line:
            mean = extract_mean(curr_line)
            std = extract_std(curr_line)
        elif 'ParSum1' in curr_line:
            z1 = x
            print("par sum1")
        elif 'ParSum2' in curr_line:
            z2 = x
            x = z1
        elif 'ParSumComplete' in curr_line:
            x = tf.add(z2,x)
        elif 'ParSumReLU' in curr_line:
            x = tf.nn.relu(tf.add(z2,x))
        elif 'SkipNet1' in curr_line:
            y = x
            print("skip net1")
        elif 'SkipNet2' in curr_line:
            print("skip net2")
            #y = tf.placeholder(tf.float64, [in_len], name = "y")
            tmp = x
            x = y
            y = tmp
        elif 'SkipCat' in curr_line:
            print("skip concatenation ",x.shape[0],x.shape[1],y.shape[0],y.shape[1])
            x = tf.concat([y,x],1)
        elif curr_line in ["ReLU", "Sigmoid", "Tanh", "Affine"]:
            print(curr_line)
            W = None
            if (last_layer in ["Conv2D", "ParSumComplete", "ParSumReLU"]) and is_trained_with_pytorch:
                W = myConst(permutation(parseVec(net), h, w, c).transpose())
            else:
                W = myConst(parseVec(net).transpose())
            b = parseVec(net)
            #b = myConst(b.reshape([1, numel(b)]))
            b = myConst(b)
            if(curr_line=="Affine"):
                x = tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b)
            elif(curr_line=="ReLU"):
                x = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))
            elif(curr_line=="Sigmoid"):
                x = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))
            else:
                x = tf.nn.tanh(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))
            print("\tOutShape: ", x.shape)
            print("\tWShape: ", W.shape)
            print("\tBShape: ", b.shape)
        elif curr_line == "MaxPooling2D":
            maxpool_line = net.readline()[:-1]
            if 'stride' in maxpool_line:
                args = runRepl(maxpool_line, ["input_shape" , "pool_size", "stride", "padding"])
                stride = [1] + args['stride'] + [1]
            else:
                args = runRepl(maxpool_line, ["input_shape" , "pool_size"])
                stride = [1] + args['pool_size'] + [1]
            if("padding" in maxpool_line):
                if(args["padding"]==1):
                    padding_arg = "SAME"
                else:
                    padding_arg = "VALID"
            else:
                padding_arg = "VALID"
            ksize =  [1] + args['pool_size'] + [1]
            print("MaxPool", args)

            x = tf.nn.max_pool(tf.reshape(x, [1] + args["input_shape"]), padding=padding_arg, strides=stride, ksize=ksize)
            print("\tOutShape: ", x.shape)
        elif curr_line == "Conv2D":
            is_conv = True
            line = net.readline()
            args = None
            #print(line[-10:-3])
            start = 0
            if("ReLU" in line):
                start = 5
            elif("Sigmoid" in line):
                start = 8
            elif("Tanh" in line):
                start = 5
            elif("Affine" in line):
                start = 7
            if 'padding' in line:
                args =  runRepl(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
            else:
                args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size"])

            W = myConst(parseVec(net))
            print("W shape", W.shape)
            #W = myConst(permutation(parseVec(net), h, w, c).transpose())
            b = None
            if("padding" in line):
                if(args["padding"]>=1):
                    padding_arg = "SAME"
                else:
                    padding_arg = "VALID"
            else:
                padding_arg = "VALID"

            if("stride" in line):
                stride_arg = [1] + args["stride"] + [1]
            else:
                stride_arg = [1,1,1,1]

            x = tf.nn.conv2d(tf.reshape(x, [1] + args["input_shape"]), filter=W, strides=stride_arg, padding=padding_arg)

            b = myConst(parseVec(net))
            h, w, c = [int(i) for i in x.shape ][1:]
            print("Conv2D", args, "W.shape:",W.shape, "b.shape:", b.shape)
            print("\tOutShape: ", x.shape)
            if("ReLU" in line):
                x = tf.nn.relu(tf.nn.bias_add(x, b))
            elif("Sigmoid" in line):
                x = tf.nn.sigmoid(tf.nn.bias_add(x, b))
            elif("Tanh" in line):
                x = tf.nn.tanh(tf.nn.bias_add(x, b))
            elif("Affine" in line):
                x = tf.nn.bias_add(x, b)
            else:
                raise Exception("Unsupported activation: ", curr_line)
        elif curr_line == "":
            break
        else:
            raise Exception("Unsupported Operation: ", curr_line)
        last_layer = curr_line

    model = x
    return model, is_conv, mean, std


def read_onnx_net(net_file):
    onnx_model = onnx.load(net_file)
    onnx.checker.check_model(onnx_model)

    is_conv = False

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            is_conv = True
            break

    return onnx_model, is_conv
