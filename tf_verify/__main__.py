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


import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import torch
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import spatial
from copy import deepcopy
from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *
from pprint import pprint
# if config.domain=='gpupoly' or config.domain=='refinegpupoly':
from refine_gpupoly import *
from utils import parse_vnn_lib_prop, translate_output_constraints, translate_input_to_box

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname



def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return list(boxes)


def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')


def normalize(image, means, stds, dataset):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        
        is_gpupoly = (domain=='gpupoly' or domain=='refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
            #for i in range(1024):
            #    image[i*3] = tmp[i]
            #    image[i*3+1] = tmp[i+1024]
            #    image[i*3+2] = tmp[i+2048]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1


def normalize_plane(plane, mean, std, channel, is_constant):
    plane_ = plane.clone()

    if is_constant:
        plane_ -= mean[channel]

    plane_ /= std[channel]

    return plane_


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    # normalization taken out of the network
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]


def model_predict(base, input):
    if is_onnx:
        pred = base.run(input)
    else:
        pred = base.run(base.graph.get_operation_by_name(model.op.name).outputs[0], {base.graph.get_operations()[0].name + ':0': input})
    return pred


def estimate_grads(specLB, specUB, dim_samples=3, input_shape=[1]):
    # Estimate gradients using central difference quotient and average over dim_samples+1 in the range of the input bounds
    # Very computationally costly
    specLB = np.array(specLB, dtype=np.float32)
    specUB = np.array(specUB, dtype=np.float32)
    inputs = [(((dim_samples - i) * specLB + i * specUB) / dim_samples).reshape(*input_shape) for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB))

    # refactor this out of this method
    if is_onnx:
        runnable = rt.prepare(model, 'CPU')
    elif sess is None:
        config = tf.ConfigProto(device_count={'GPU': 0})
        runnable = tf.Session(config=config)
    else:
        runnable = sess

    for sample in range(dim_samples + 1):
        pred = model_predict(runnable, inputs[sample])

        for index in range(len(specLB)):
            if sample < dim_samples:
                l_input = [m if i != index else u for i, m, u in zip(range(len(specLB)), inputs[sample], inputs[sample+1])]
                l_input = np.array(l_input, dtype=np.float32)
                l_i_pred = model_predict(runnable, l_input)
            else:
                l_i_pred = pred
            if sample > 0:
                u_input = [m if i != index else l for i, m, l in zip(range(len(specLB)), inputs[sample], inputs[sample-1])]
                u_input = np.array(u_input, dtype=np.float32)
                u_i_pred = model_predict(runnable, u_input)
            else:
                u_i_pred = pred
            diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
            diffs[index] += diff
    return diffs / dim_samples



progress = 0.0
def print_progress(depth):
    if config.debug:
        global progress, rec_start
        progress += np.power(2.,-depth)
        sys.stdout.write('\r%.10f percent, %.02f s' % (100 * progress, time.time()-rec_start))


def acasxu_recursive(specLB, specUB, max_depth=10, depth=0):
    hold,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
    global failed_already
    if hold:
        print_progress(depth)
        return hold
    elif depth >= max_depth:
        if failed_already.value and config.complete:
            verified_flag, adv_examples, _ = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            print_progress(depth)
            if verified_flag == False:
                if adv_examples!=None:
                    #print("adv image ", adv_image)
                    for adv_image in adv_examples:
                        hold,_,nlb,nub,_,_ = eran.analyze_box(adv_image, adv_image, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                        #print("hold ", hold, "domain", domain)
                        if hold == False:
                            print("property violated at ", adv_image, "output_score", nlb[-1])
                            failed_already.value = 0
                            break
            return verified_flag
        else:
            return False
    else:
        # grads = estimate_grads(specLB, specUB, input_shape=eran.input_shape)
        # # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
        # smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])

        #start = time.time()
        nn.set_last_weights(constraints)
        grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)
        smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]

        index = np.argmax(smears)
        m = (specLB[index]+specUB[index])/2

        result = failed_already.value and acasxu_recursive(specLB, [ub if i != index else m for i, ub in enumerate(specUB)], max_depth, depth + 1)
        result = failed_already.value and result and acasxu_recursive([lb if i != index else m for i, lb in enumerate(specLB)], specUB, max_depth, depth + 1)
        return result



def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        if config.subset == None:
            try:
                csvfile = open('../data/{}_test_full.csv'.format(dataset), 'r')
            except:
                csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
                print("Only the first 100 samples are available.")
        else:
            filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
            csvfile = open(filename, 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
parser.add_argument('--zonotope', type=str, default=config.zonotope, help='file to specify the zonotope matrix')
parser.add_argument('--subset', type=str, default=config.subset, help='suffix of the file to specify the subset of the test dataset to use')
parser.add_argument('--target', type=str, default=config.target, help='file specify the targets for the attack')
parser.add_argument('--epsfile', type=str, default=config.epsfile, help='file specify the epsilons for the L_oo attack')
parser.add_argument('--vnn_lib_spec', type=str, default=config.vnn_lib_spec, help='VNN_LIB spec file, defining input and output constraints')
parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
parser.add_argument('--timeout_final_lp', type=float, default=config.timeout_final_lp,  help='timeout for the final LP solver')
parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
parser.add_argument('--timeout_final_milp', type=float, default=config.timeout_final_lp,  help='timeout for the final MILP solver')
parser.add_argument('--timeout_complete', type=float, default=None,  help='Cumulative timeout for the complete verifier, superseeds timeout_final_milp if set')
parser.add_argument('--max_milp_neurons', type=int, default=config.max_milp_neurons,  help='number of layers to encode using MILP.')
parser.add_argument('--partial_milp', type=int, default=config.partial_milp,  help='Maximum number of neurons to use for partial MILP encoding')

parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
parser.add_argument('--n_milp_refine', type=int, default=config.n_milp_refine, help='Number of milp refined layers')
parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
parser.add_argument('--geometric_config', type=str, default=config.geometric_config, help='config location')
parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')
parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack')
parser.add_argument('--geometric', '-g', dest='geometric', default=config.geometric, action='store_true', help='Whether to do geometric analysis')
parser.add_argument('--input_box', default=config.input_box,  help='input box to use')
parser.add_argument('--output_constraints', default=config.output_constraints, help='custom output constraints to check')
parser.add_argument('--normalized_region', type=str2bool, default=config.normalized_region, help='Whether to normalize the adversarial region')
parser.add_argument('--spatial', action='store_true', default=config.spatial, help='whether to do vector field analysis')
parser.add_argument('--t-norm', type=str, default=config.t_norm, help='vector field norm (1, 2, or inf)')
parser.add_argument('--delta', type=float, default=config.delta, help='vector field displacement magnitude')
parser.add_argument('--gamma', type=float, default=config.gamma, help='vector field smoothness constraint')
parser.add_argument('--k', type=int, default=config.k, help='refine group size')
parser.add_argument('--s', type=int, default=config.s, help='refine group sparsity parameter')
parser.add_argument('--quant_step', type=float, default=config.quant_step, help='Quantization step for quantized networks')
parser.add_argument("--approx_k", type=str2bool, default=config.approx_k, help="Use approximate fast k neuron constraints")


# Logging options
parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')


args = parser.parse_args()
for k, v in vars(args).items():
    setattr(config, k, v)
# if args.timeout_complete is not None:
#     raise DeprecationWarning("'--timeout_complete' is depreciated. Use '--timeout_final_milp' instead")
config.json = vars(args)
pprint(config.json)

if config.specnumber and not config.input_box and not config.output_constraints:
    config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
    config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

assert config.netname, 'a network has to be provided for analysis.'

#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

netname = config.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = config.epsilon
#assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = config.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = config.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not config.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly', 'refinegpupoly'], "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

dataset = config.dataset

if zonotope_bool==False:
   assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

mean = 0
std = 0

complete = (config.complete==True)

if(dataset=='acasxu'):
    print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
else:
    print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

sess = None
if is_saved_tf_model or is_pb_file:
    netfolder = os.path.dirname(netname)

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    if is_saved_tf_model:
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    else:
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')

    eran = ERAN(model, sess)

else:
    if(zonotope_bool==True):
        num_pixels = len(zonotope)
    elif(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    if is_onnx:
        model, is_conv = read_onnx_net(netname)
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        if is_onnx:
            translator = ONNXTranslator(model, True)
        else:
            translator = TFTranslator(model)
        operations, resources = translator.translate()
        optimizer = Optimizer(operations, resources)
        nn = layers()
        network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn) 
    else:    
        eran = ERAN(model, is_onnx=is_onnx)

if not is_trained_with_pytorch:
    if dataset == 'mnist' and not config.geometric:
        means = [0]
        stds = [1]
    elif dataset == 'acasxu':
        means = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    elif dataset == "cifar10":
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2023, 0.1994, 0.2010]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]

is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std

os.sched_setaffinity(0,cpu_affinity)

correctly_classified_images = 0
verified_images = 0
unsafe_images = 0
cum_time = 0

if config.vnn_lib_spec is not None:
    # input and output constraints in homogenized representation x >= C_lb * [x_0, eps, 1]; C_out [y, 1] >= 0
    C_lb, C_ub, C_out = parse_vnn_lib_prop(config.vnn_lib_spec)
    constraints = translate_output_constraints(C_out)
    boxes = translate_input_to_box(C_lb, C_ub, x_0=None, eps=None, domain_bounds=None)
else:
    if config.output_constraints:
        constraints = get_constraints_from_file(config.output_constraints)
    else:
        constraints = None

    if dataset and config.input_box is None:
        tests = get_tests(dataset, config.geometric)
    else:
        tests = open(config.input_box, 'r').read()
        boxes = parse_input_box(tests)

def init(args):
    global failed_already
    failed_already = args

if dataset=='acasxu':
    use_parallel_solve = True
    if config.debug:
        print('Constraints: ', constraints)
    total_start = time.time()
    for box_index, box in enumerate(boxes):
        specLB = [interval[0] for interval in box]
        specUB = [interval[1] for interval in box]
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)

        e = None
        holds = True
        x_adex = None
        adex_holds = True

        rec_start = time.time()
        # start = time.time()

        verified_flag, nn, nlb, nub, _ , x_adex = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
        if not verified_flag and x_adex is not None:
            adex_holds, _, _, _, _, _ = eran.analyze_box(x_adex, x_adex, "deeppoly", config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)

        if not verified_flag and adex_holds:
            # expensive min/max gradient calculation
            nn.set_last_weights(constraints)
            grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)

            smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
            split_multiple = 20 / np.sum(smears)

            num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
            step_size = []
            for i in range(5):
                if num_splits[i]==0:
                    num_splits[i] = 1
                step_size.append((specUB[i]-specLB[i])/num_splits[i])
            #sorted_indices = np.argsort(widths)
            #input_to_split = sorted_indices[0]
            #print("input to split ", input_to_split)

            #step_size = widths/num_splits
            #print("step size", step_size,num_splits)
            start_val = np.copy(specLB)
            end_val = np.copy(specUB)

            # _,nn,_,_,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
            #complete_list = []
            multi_bounds = []
            for i in range(num_splits[0]):
                if not holds: break
                specLB[0] = start_val[0] + i*step_size[0]
                specUB[0] = np.fmin(end_val[0],start_val[0]+ (i+1)*step_size[0])

                for j in range(num_splits[1]):
                    if not holds: break
                    specLB[1] = start_val[1] + j*step_size[1]
                    specUB[1] = np.fmin(end_val[1],start_val[1]+ (j+1)*step_size[1])

                    for k in range(num_splits[2]):
                        if not holds: break
                        specLB[2] = start_val[2] + k*step_size[2]
                        specUB[2] = np.fmin(end_val[2],start_val[2]+ (k+1)*step_size[2])

                        for l in range(num_splits[3]):
                            if not holds: break
                            specLB[3] = start_val[3] + l*step_size[3]
                            specUB[3] = np.fmin(end_val[3],start_val[3]+ (l+1)*step_size[3])
                            for m in range(num_splits[4]):
                                specLB[4] = start_val[4] + m*step_size[4]
                                specUB[4] = np.fmin(end_val[4],start_val[4]+ (m+1)*step_size[4])

                                if use_parallel_solve:
                                    # add bounds to input for multiprocessing map
                                    multi_bounds.append((specLB.copy(), specUB.copy()))
                                else:
                                    # --- VERSION WITHOUT MULTIPROCESSING ---
                                    holds, _, nlb, nub, _, x_adex = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)

                                    if not holds:
                                        if x_adex is not None:
                                            adex_holds, _, _, _, _, _ = eran.analyze_box(x_adex, x_adex, "deeppoly", config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                                            if not adex_holds:
                                                verified_flag = False
                                                break
                                        if complete:
                                            holds, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                                            #complete_list.append((i,j,k,l,m))
                                            if not holds:
                                                verified_flag = False
                                                break
                                        else:
                                            verified_flag = False
                                            break
                                    if config.debug:
                                       sys.stdout.write('\rsplit %i, %i, %i, %i, %i %.02f sec\n' % (i, j, k, l, m, time.time()-rec_start))

            #print(time.time() - rec_start, "seconds")
            #print("LENGTH ", len(multi_bounds))
            if use_parallel_solve:
                failed_already = Value('i', 1)
                try:
                    with Pool(processes=10, initializer=init, initargs=(failed_already,)) as pool:
                        res = pool.starmap(acasxu_recursive, multi_bounds)

                    if all(res):
                        verified_flag = True
                    else:
                        verified_flag = False
                except Exception as ex:
                    verified_flag = False
                    e = ex

        ver_str = "Verified" if verified_flag else "Failed"
        if not adex_holds:
            ver_str += " with counterexample"
        if e is None:
            print("AcasXu property", config.specnumber, f"{ver_str} for Box", box_index, "out of", len(boxes))
        else:
            print("AcasXu property", config.specnumber, "Failed for Box", box_index, "out of", len(boxes), "because of an exception ", e)

        print(time.time() - rec_start, "seconds")
    print("Total time needed:", time.time() - total_start, "seconds")

elif zonotope_bool:
    perturbed_label, nn, nlb, nub,_,_ = eran.analyze_zonotope(zonotope, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
    print("nlb ",nlb[-1])
    print("nub ",nub[-1])
    if(perturbed_label!=-1):
        print("Verified")
    elif(complete==True):
        constraints = get_constraints_for_dominant_label(perturbed_label, 10)
        verified_flag, adv_image, _ = verify_network_with_milp(nn, zonotope, [], nlb, nub, constraints)
        if(verified_flag==True):
            print("Verified")
        else:
            print("Failed")
    else:
         print("Failed")


elif config.geometric:
    from geometric_constraints import *
    total, attacked, standard_correct, tot_time = 0, 0, 0, 0
    correct_box, correct_poly = 0, 0
    cver_box, cver_poly = [], []
    if config.geometric_config:
        transform_attack_container = get_transform_attack_container(config.geometric_config)
        for i, test in enumerate(tests):
            if config.from_test and i < config.from_test:
                continue

            if config.num_tests is not None and i >= config.num_tests:
                break
            set_transform_attack_for(transform_attack_container, i, config.attack, config.debug)
            attack_params = get_attack_params(transform_attack_container)
            attack_images = get_attack_images(transform_attack_container)
            print('Test {}:'.format(i))

            image = np.float64(test[1:])
            if config.dataset == 'mnist' or config.dataset == 'fashion':
                n_rows, n_cols, n_channels = 28, 28, 1
            else:
                n_rows, n_cols, n_channels = 32, 32, 3

            spec_lb = np.copy(image)
            spec_ub = np.copy(image)

            normalize(spec_lb, means, stds, config.dataset)
            normalize(spec_ub, means, stds, config.dataset)

            label, nn, nlb, nub,_,_ = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', config.timeout_lp, config.timeout_milp,
                                                   config.use_default_heuristic)
            print('Label: ', label)

            begtime = time.time()
            if label != int(test[0]):
                print('Label {}, but true label is {}, skipping...'.format(label, int(test[0])))
                print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))
                continue
            else:
                standard_correct += 1
                print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))

            dim = n_rows * n_cols * n_channels

            ok_box, ok_poly = True, True
            k = config.num_params + 1 + 1 + dim

            attack_imgs, checked, attack_pass = [], [], 0
            cex_found = False
            if config.attack:
                for j in tqdm(range(0, len(attack_params))):
                    params = attack_params[j]
                    values = np.array(attack_images[j])

                    attack_lb = values[::2]
                    attack_ub = values[1::2]

                    normalize(attack_lb, means, stds, config.dataset)
                    normalize(attack_ub, means, stds, config.dataset)
                    attack_imgs.append((params, attack_lb, attack_ub))
                    checked.append(False)

                    predict_label, _, _, _, _, _ = eran.analyze_box(
                        attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                        config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                    if predict_label != int(test[0]):
                        print('counter-example, params: ', params, ', predicted label: ', predict_label)
                        cex_found = True
                        break
                    else:
                        attack_pass += 1
            print('tot attacks: ', len(attack_imgs))

            lines = get_transformations(transform_attack_container)
            print('Number of lines: ', len(lines))
            assert len(lines) % k == 0

            spec_lb = np.zeros(config.num_params + dim)
            spec_ub = np.zeros(config.num_params + dim)

            expr_size = config.num_params
            lexpr_cst, uexpr_cst = [], []
            lexpr_weights, uexpr_weights = [], []
            lexpr_dim, uexpr_dim = [], []

            ver_chunks_box, ver_chunks_poly, tot_chunks = 0, 0, 0

            for i, line in enumerate(lines):
                if i % k < config.num_params:
                    # read specs for the parameters
                    values = line
                    assert len(values) == 2
                    param_idx = i % k
                    spec_lb[dim + param_idx] = values[0]
                    spec_ub[dim + param_idx] = values[1]
                    if config.debug:
                        print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
                elif i % k == config.num_params:
                    # read interval bounds for image pixels
                    values = line
                    spec_lb[:dim] = values[::2]
                    spec_ub[:dim] = values[1::2]
                    # if config.debug:
                    #     show_ascii_spec(spec_lb, spec_ub)
                elif i % k < k - 1:
                    # read polyhedra constraints for image pixels
                    tokens = line
                    assert len(tokens) == 2 + 2 * config.num_params

                    bias_lower, weights_lower = tokens[0], tokens[1:1 + config.num_params]
                    bias_upper, weights_upper = tokens[config.num_params + 1], tokens[2 + config.num_params:]

                    assert len(weights_lower) == config.num_params
                    assert len(weights_upper) == config.num_params

                    lexpr_cst.append(bias_lower)
                    uexpr_cst.append(bias_upper)
                    for j in range(config.num_params):
                        lexpr_dim.append(dim + j)
                        uexpr_dim.append(dim + j)
                        lexpr_weights.append(weights_lower[j])
                        uexpr_weights.append(weights_upper[j])
                else:
                    assert (len(line) == 0)
                    for p_idx in range(config.num_params):
                        lexpr_cst.append(spec_lb[dim + p_idx])
                        for l in range(config.num_params):
                            lexpr_weights.append(0)
                            lexpr_dim.append(dim + l)
                        uexpr_cst.append(spec_ub[dim + p_idx])
                        for l in range(config.num_params):
                            uexpr_weights.append(0)
                            uexpr_dim.append(dim + l)
                    normalize(spec_lb[:dim], means, stds, config.dataset)
                    normalize(spec_ub[:dim], means, stds, config.dataset)
                    normalize_poly(config.num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights,
                                   uexpr_dim, means, stds, config.dataset)

                    for attack_idx, (attack_params, attack_lb, attack_ub) in enumerate(attack_imgs):
                        ok_attack = True
                        for j in range(num_pixels):
                            low, up = lexpr_cst[j], uexpr_cst[j]
                            for idx in range(config.num_params):
                                low += lexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                up += uexpr_weights[j * config.num_params + idx] * attack_params[idx]
                            if low > attack_lb[j] + EPS or attack_ub[j] > up + EPS:
                                ok_attack = False
                        if ok_attack:
                            checked[attack_idx] = True
                            # print('checked ', attack_idx)
                    if config.debug:
                        print('Running the analysis...')

                    t_begin = time.time()
                    perturbed_label_poly, _, _, _, _, _ = eran.analyze_box(
                        spec_lb, spec_ub, 'deeppoly',
                        config.timeout_lp, config.timeout_milp, config.use_default_heuristic, None,
                        lexpr_weights, lexpr_cst, lexpr_dim,
                        uexpr_weights, uexpr_cst, uexpr_dim,
                        expr_size)
                    perturbed_label_box, _, _, _, _, _ = eran.analyze_box(
                        spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                        config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                    t_end = time.time()

                    print('DeepG: ', perturbed_label_poly, '\tInterval: ', perturbed_label_box, '\tlabel: ', label,
                          '[Time: %.4f]' % (t_end - t_begin))

                    tot_chunks += 1
                    if perturbed_label_box != label:
                        ok_box = False
                    else:
                        ver_chunks_box += 1

                    if perturbed_label_poly != label:
                        ok_poly = False
                    else:
                        ver_chunks_poly += 1

                    lexpr_cst, uexpr_cst = [], []
                    lexpr_weights, uexpr_weights = [], []
                    lexpr_dim, uexpr_dim = [], []

            total += 1
            if ok_box:
                correct_box += 1
            if ok_poly:
                correct_poly += 1
            if cex_found:
                assert (not ok_box) and (not ok_poly)
                attacked += 1
            cver_poly.append(ver_chunks_poly / float(tot_chunks))
            cver_box.append(ver_chunks_box / float(tot_chunks))
            tot_time += time.time() - begtime

            print('Verified[box]: {}, Verified[poly]: {}, CEX found: {}'.format(ok_box, ok_poly, cex_found))
            assert not cex_found or not ok_box, 'ERROR! Found counter-example, but image was verified with box!'
            assert not cex_found or not ok_poly, 'ERROR! Found counter-example, but image was verified with poly!'


    else:
        for i, test in enumerate(tests):
            if config.from_test and i < config.from_test:
                continue

            if config.num_tests is not None and i >= config.num_tests:
                break

            attacks_file = os.path.join(config.data_dir, 'attack_{}.csv'.format(i))
            print('Test {}:'.format(i))

            image = np.float64(test[1:])
            if config.dataset == 'mnist' or config.dataset == 'fashion':
                n_rows, n_cols, n_channels = 28, 28, 1
            else:
                n_rows, n_cols, n_channels = 32, 32, 3

            spec_lb = np.copy(image)
            spec_ub = np.copy(image)

            normalize(spec_lb, means, stds, config.dataset)
            normalize(spec_ub, means, stds, config.dataset)
            
            label, nn, nlb, nub, _, _ = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', config.timeout_lp, config.timeout_milp,
                                                   config.use_default_heuristic)
            print('Label: ', label)

            begtime = time.time()
            if label != int(test[0]):
                print('Label {}, but true label is {}, skipping...'.format(label, int(test[0])))
                print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))
                continue
            else:
                standard_correct += 1
                print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))

            dim = n_rows * n_cols * n_channels

            ok_box, ok_poly = True, True
            k = config.num_params + 1 + 1 + dim

            attack_imgs, checked, attack_pass = [], [], 0
            cex_found = False
            if config.attack:
                with open(attacks_file, 'r') as fin:
                    lines = fin.readlines()
                    for j in tqdm(range(0, len(lines), config.num_params + 1)):
                        params = [float(line[:-1]) for line in lines[j:j + config.num_params]]
                        tokens = lines[j + config.num_params].split(',')
                        values = np.array(list(map(float, tokens)))

                        attack_lb = values[::2]
                        attack_ub = values[1::2]

                        normalize(attack_lb, means, stds, config.dataset)
                        normalize(attack_ub, means, stds, config.dataset)
                        attack_imgs.append((params, attack_lb, attack_ub))
                        checked.append(False)

                        predict_label, _, _, _, _, _ = eran.analyze_box(
                            attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                            config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                        if predict_label != int(test[0]):
                            print('counter-example, params: ', params, ', predicted label: ', predict_label)
                            cex_found = True
                            break
                        else:
                            attack_pass += 1
            print('tot attacks: ', len(attack_imgs))
            specs_file = os.path.join(config.data_dir, '{}.csv'.format(i))
            with open(specs_file, 'r') as fin:
                lines = fin.readlines()
                print('Number of lines: ', len(lines))
                assert len(lines) % k == 0

                spec_lb = np.zeros(config.num_params + dim)
                spec_ub = np.zeros(config.num_params + dim)

                expr_size = config.num_params
                lexpr_cst, uexpr_cst = [], []
                lexpr_weights, uexpr_weights = [], []
                lexpr_dim, uexpr_dim = [], []

                ver_chunks_box, ver_chunks_poly, tot_chunks = 0, 0, 0

                for i, line in enumerate(lines):
                    if i % k < config.num_params:
                        # read specs for the parameters
                        values = np.array(list(map(float, line[:-1].split(' '))))
                        assert values.shape[0] == 2
                        param_idx = i % k
                        spec_lb[dim + param_idx] = values[0]
                        spec_ub[dim + param_idx] = values[1]
                        if config.debug:
                            print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
                    elif i % k == config.num_params:
                        # read interval bounds for image pixels
                        values = np.array(list(map(float, line[:-1].split(','))))
                        spec_lb[:dim] = values[::2]
                        spec_ub[:dim] = values[1::2]
                        # if config.debug:
                        #     show_ascii_spec(spec_lb, spec_ub)
                    elif i % k < k - 1:
                        # read polyhedra constraints for image pixels
                        tokens = line[:-1].split(' ')
                        assert len(tokens) == 2 + 2 * config.num_params + 1

                        bias_lower, weights_lower = float(tokens[0]), list(map(float, tokens[1:1 + config.num_params]))
                        assert tokens[config.num_params + 1] == '|'
                        bias_upper, weights_upper = float(tokens[config.num_params + 2]), list(
                            map(float, tokens[3 + config.num_params:]))

                        assert len(weights_lower) == config.num_params
                        assert len(weights_upper) == config.num_params

                        lexpr_cst.append(bias_lower)
                        uexpr_cst.append(bias_upper)
                        for j in range(config.num_params):
                            lexpr_dim.append(dim + j)
                            uexpr_dim.append(dim + j)
                            lexpr_weights.append(weights_lower[j])
                            uexpr_weights.append(weights_upper[j])
                    else:
                        assert (line == 'SPEC_FINISHED\n')
                        for p_idx in range(config.num_params):
                            lexpr_cst.append(spec_lb[dim + p_idx])
                            for l in range(config.num_params):
                                lexpr_weights.append(0)
                                lexpr_dim.append(dim + l)
                            uexpr_cst.append(spec_ub[dim + p_idx])
                            for l in range(config.num_params):
                                uexpr_weights.append(0)
                                uexpr_dim.append(dim + l)
                        normalize(spec_lb[:dim], means, stds, config.dataset)
                        normalize(spec_ub[:dim], means, stds, config.dataset)
                        normalize_poly(config.num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights,
                                       uexpr_dim, means, stds, config.dataset)

                        for attack_idx, (attack_params, attack_lb, attack_ub) in enumerate(attack_imgs):
                            ok_attack = True
                            for j in range(num_pixels):
                                low, up = lexpr_cst[j], uexpr_cst[j]
                                for idx in range(config.num_params):
                                    low += lexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                    up += uexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                if low > attack_lb[j] + EPS or attack_ub[j] > up + EPS:
                                    ok_attack = False
                            if ok_attack:
                                checked[attack_idx] = True
                                # print('checked ', attack_idx)
                        if config.debug:
                            print('Running the analysis...')

                        t_begin = time.time()
                        perturbed_label_poly, _, _, _ , _, _ = eran.analyze_box(
                            spec_lb, spec_ub, 'deeppoly',
                            config.timeout_lp, config.timeout_milp, config.use_default_heuristic, None,
                            lexpr_weights, lexpr_cst, lexpr_dim,
                            uexpr_weights, uexpr_cst, uexpr_dim,
                            expr_size)
                        perturbed_label_box, _, _, _, _, _ = eran.analyze_box(
                            spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                            config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                        t_end = time.time()

                        print('DeepG: ', perturbed_label_poly, '\tInterval: ', perturbed_label_box, '\tlabel: ', label,
                              '[Time: %.4f]' % (t_end - t_begin))

                        tot_chunks += 1
                        if perturbed_label_box != label:
                            ok_box = False
                        else:
                            ver_chunks_box += 1

                        if perturbed_label_poly != label:
                            ok_poly = False
                        else:
                            ver_chunks_poly += 1

                        lexpr_cst, uexpr_cst = [], []
                        lexpr_weights, uexpr_weights = [], []
                        lexpr_dim, uexpr_dim = [], []

            total += 1
            if ok_box:
                correct_box += 1
            if ok_poly:
                correct_poly += 1
            if cex_found:
                assert (not ok_box) and (not ok_poly)
                attacked += 1
            cver_poly.append(ver_chunks_poly / float(tot_chunks))
            cver_box.append(ver_chunks_box / float(tot_chunks))
            tot_time += time.time() - begtime

            print('Verified[box]: {}, Verified[poly]: {}, CEX found: {}'.format(ok_box, ok_poly, cex_found))
            assert not cex_found or not ok_box, 'ERROR! Found counter-example, but image was verified with box!'
            assert not cex_found or not ok_poly, 'ERROR! Found counter-example, but image was verified with poly!'

    print('Attacks found: %.2f percent, %d/%d' % (100.0 * attacked / total, attacked, total))
    print('[Box]  Provably robust: %.2f percent, %d/%d' % (100.0 * correct_box / total, correct_box, total))
    print('[Poly] Provably robust: %.2f percent, %d/%d' % (100.0 * correct_poly / total, correct_poly, total))
    print('Empirically robust: %.2f percent, %d/%d' % (100.0 * (total - attacked) / total, total - attacked, total))
    print('[Box]  Average chunks verified: %.2f percent' % (100.0 * np.mean(cver_box)))
    print('[Poly]  Average chunks verified: %.2f percent' % (100.0 * np.mean(cver_poly)))
    print('Average time: ', tot_time / total)

elif config.input_box is not None:
    boxes = parse_input_box(tests)
    index = 1
    correct = 0
    for box in boxes:
        specLB = [interval[0] for interval in box]
        specUB = [interval[1] for interval in box]
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)
        hold, nn, nlb, nub,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
        if hold:
            print('constraints hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))
            correct += 1
        else:
            print('constraints do NOT hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))

        index += 1

    print('constraints hold for ' + str(correct) + ' out of ' + str(sum([1 for b in boxes])) + ' boxes')

elif config.spatial:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.dataset in ['mnist', 'fashion']:
        height, width, channels = 28, 28, 1
    else:
        height, width, channels = 32, 32, 3

    for idx, test in enumerate(tests):

        if idx < config.from_test:
            continue

        if (config.num_tests is not None) and (config.from_test + config.num_tests == idx):
            break

        image = torch.from_numpy(
            np.float64(test[1:len(test)]) / np.float64(255)
        ).reshape(1, height, width, channels).permute(0, 3, 1, 2).to(device)
        label = np.int(test[0])

        specLB = image.clone().permute(0, 2, 3, 1).flatten().cpu()
        specUB = image.clone().permute(0, 2, 3, 1).flatten().cpu()
        
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)

        predicted_label, nn, nlb, nub, _, _ = eran.analyze_box(
            specLB=specLB, specUB=specUB, domain=init_domain(domain), 
            timeout_lp=config.timeout_lp, timeout_milp=config.timeout_milp, 
            use_default_heuristic=config.use_default_heuristic
        )

        print(f'concrete {nlb[-1]}')

        if label != predicted_label:
            print(f'img {idx} not considered, correct_label {label}, classified label {predicted_label}')
            continue

        correctly_classified_images += 1
        start = time.time()

        transformer = getattr(
            spatial, f'T{config.t_norm.capitalize()}NormTransformer'
        )(image, config.delta)
        box_lb, box_ub = transformer.box_constraints()

        lower_bounds = box_lb.permute(0, 2, 3, 1).flatten()
        upper_bounds = box_ub.permute(0, 2, 3, 1).flatten()

        normalize(lower_bounds, means, stds, dataset)
        normalize(upper_bounds, means, stds, dataset)

        specLB, specUB = lower_bounds.clone(), upper_bounds.clone()
        LB_N0, UB_N0 = lower_bounds.clone(), upper_bounds.clone()

        expr_size = 0
        lexpr_weights = lexpr_cst = lexpr_dim = None
        uexpr_weights = uexpr_cst = uexpr_dim = None
        lower_planes = upper_planes = None
        deeppoly_spatial_constraints = milp_spatial_constraints = None

        if config.gamma < float('inf'):

            expr_size = 2
            lower_planes, upper_planes = list(), list()
            lexpr_weights, lexpr_cst, lexpr_dim = list(), list(), list()
            uexpr_weights, uexpr_cst, uexpr_dim = list(), list(), list()

            linear_lb, linear_ub = transformer.linear_constraints()

            for channel in range(image.shape[1]):
                lb_a, lb_b, lb_c = linear_lb[channel]
                ub_a, ub_b, ub_c = linear_ub[channel]

                linear_lb[channel][0] = normalize_plane(
                    lb_a, means, stds, channel, is_constant=True
                )
                linear_lb[channel][1] = normalize_plane(
                    lb_b, means, stds, channel, is_constant=False
                )
                linear_lb[channel][2] = normalize_plane(
                    lb_c, means, stds, channel, is_constant=False
                )

                linear_ub[channel][0] = normalize_plane(
                    ub_a, means, stds, channel, is_constant=True
                )
                linear_ub[channel][1] = normalize_plane(
                    ub_b, means, stds, channel, is_constant=False
                )
                linear_ub[channel][2] = normalize_plane(
                    ub_c, means, stds, channel, is_constant=False
                )

            for i in range(3):
                lower_planes.append(
                    torch.cat(
                        [
                            linear_lb[channel][i].unsqueeze(-1)
                            for channel in range(image.shape[1])
                        ], dim=-1
                    ).flatten().tolist()
                )
                upper_planes.append(
                    torch.cat(
                        [
                            linear_ub[channel][i].unsqueeze(-1)
                            for channel in range(image.shape[1])
                        ], dim=-1
                    ).flatten().tolist()
                )

            deeppoly_spatial_constraints = {'gamma': config.gamma}

            for key, val in transformer.flow_constraint_pairs.items():
                deeppoly_spatial_constraints[key] = val.cpu()

            milp_spatial_constraints = {
                'delta': config.delta, 'gamma': config.gamma, 
                'channels': image.shape[1], 'lower_planes': lower_planes, 
                'upper_planes': upper_planes,
                'add_norm_constraints': transformer.add_norm_constraints,
                'neighboring_indices': transformer.flow_constraint_pairs
            }

            num_pixels = image.flatten().shape[0]
            num_flows = 2 * num_pixels

            flows_LB = torch.full((num_flows,), -config.delta).to(device)
            flows_UB = torch.full((num_flows,), config.delta).to(device)

            specLB = torch.cat((specLB, flows_LB))
            specUB = torch.cat((specUB, flows_UB))

            lexpr_cst = deepcopy(lower_planes[0]) + flows_LB.tolist()
            uexpr_cst = deepcopy(upper_planes[0]) + flows_UB.tolist()

            lexpr_weights = [
                v for p in zip(lower_planes[1], lower_planes[2]) for v in p
            ] + torch.zeros(2 * num_flows).tolist()
            uexpr_weights = [
                v for p in zip(upper_planes[1], upper_planes[2]) for v in p
            ] + torch.zeros(2 * num_flows).tolist()

            lexpr_dim = torch.cat([
                num_pixels + torch.arange(num_flows),
                torch.zeros(2 * num_flows).long()
            ]).tolist()
            uexpr_dim = torch.cat([
                num_pixels + torch.arange(num_flows),
                torch.zeros(2 * num_flows).long()
            ]).tolist()

        perturbed_label, _, nlb, nub, failed_labels, _ = eran.analyze_box(
            specLB=specLB.cpu(), specUB=specUB.cpu(), domain=domain,
            timeout_lp=config.timeout_lp, timeout_milp=config.timeout_milp,
            use_default_heuristic=config.use_default_heuristic,
            label=label, lexpr_weights=lexpr_weights, lexpr_cst=lexpr_cst,
            lexpr_dim=lexpr_dim, uexpr_weights=uexpr_weights, 
            uexpr_cst=uexpr_cst, uexpr_dim=uexpr_dim, expr_size=expr_size,
            spatial_constraints=deeppoly_spatial_constraints
        )
        end = time.time()

        print(f'nlb {nlb[-1]} nub {nub[-1]} adv labels {failed_labels}')

        if perturbed_label == label:
            print(f'img {idx} verified {label}')
            verified_images += 1
            print(end - start, "seconds")
            continue

        if (not complete) or (domain not in ['deeppoly', 'deepzono']):
            print(f'img {idx} Failed')
            print(end - start, "seconds")
            continue

        verified_flag, adv_image, _ = verify_network_with_milp(
            nn=nn, LB_N0=LB_N0, UB_N0=UB_N0, nlb=nlb, nub=nub,
            constraints=get_constraints_for_dominant_label(
                predicted_label, failed_labels=failed_labels
            ), spatial_constraints=milp_spatial_constraints
        )

        if verified_flag:
            print(f'img {idx} Verified as Safe {label}')
            verified_images += 1
        else:
            print(f'img {idx} Failed')

        end = time.time()
        print(end - start, "seconds")

    print(f'analysis precision {verified_images} / {correctly_classified_images}')

else:
    target = []
    if config.target != None:
        targetfile = open(config.target, 'r')
        targets = csv.reader(targetfile, delimiter=',')
        for i, val in enumerate(targets):
            target = val   
   
   
    if config.epsfile != None:
        epsfile = open(config.epsfile, 'r')
        epsilons = csv.reader(epsfile, delimiter=',')
        for i, val in enumerate(epsilons):
            eps_array = val  
            
    for i, test in enumerate(tests):
        if config.from_test and i < config.from_test:
            continue

        if config.num_tests is not None and i >= config.from_test + config.num_tests:
            break
        image= np.float64(test[1:len(test)])/np.float64(255)
        specLB = np.copy(image)
        specUB = np.copy(image)
        if config.quant_step:
            specLB = np.round(specLB/config.quant_step)
            specUB = np.round(specUB/config.quant_step)
        #cifarfile = open('/home/gagandeepsi/eevbnn/input.txt', 'r')
        
        #cifarimages = csv.reader(cifarfile, delimiter=',')
        #for _, image in enumerate(cifarimages):
        #    specLB = np.float64(image)
        #specUB = np.copy(specLB)
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)


        #print("specLB ", len(specLB), "specUB ", specUB)
        is_correctly_classified = False
        start = time.time()
        if domain == 'gpupoly' or domain == 'refinegpupoly':
            #specLB = np.reshape(specLB, (32,32,3))#np.ascontiguousarray(specLB, dtype=np.double)
            #specUB = np.reshape(specUB, (32,32,3))
            #print("specLB ", specLB)
            is_correctly_classified = network.test(specLB, specUB, int(test[0]), True)
        else:
            label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
            print("concrete ", nlb[-1])
            if label == int(test[0]):
                is_correctly_classified = True
        #for number in range(len(nub)):
        #    for element in range(len(nub[number])):
        #        if(nub[number][element]<=0):
        #            print('False')
        #        else:
        #            print('True')
        if config.epsfile!= None:
            epsilon = np.float64(eps_array[i])
        
        #if(label == int(test[0])):
        if is_correctly_classified == True:
            label = int(test[0])
            perturbed_label = None
            correctly_classified_images +=1
            if config.normalized_region==True:
                specLB = np.clip(image - epsilon,0,1)
                specUB = np.clip(image + epsilon,0,1)
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)
            else:
                specLB = specLB - epsilon
                specUB = specUB + epsilon

            if config.quant_step:
                specLB = np.round(specLB/config.quant_step)
                specUB = np.round(specUB/config.quant_step)

            if config.target == None:
                prop = -1
            else:
                prop = int(target[i])

            if domain == 'gpupoly' or domain =='refinegpupoly':
                is_verified = network.test(specLB, specUB, int(test[0]))
                #print("res ", res)
                if is_verified:
                    print("img", i, "Verified", int(test[0]))
                    verified_images+=1
                elif domain == 'refinegpupoly':
                    num_outputs = len(nn.weights[-1])

                    # Matrix that computes the difference with the expected layer.
                    diffMatrix = np.delete(-np.eye(num_outputs), int(test[0]), 0)
                    diffMatrix[:, label] = 1
                    diffMatrix = diffMatrix.astype(np.float64)
                    
                    # gets the values from GPUPoly.
                    res = network.evalAffineExpr(diffMatrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)
                    
                    
                    labels_to_be_verified = []
                    var = 0
                    nn.specLB = specLB
                    nn.specUB = specUB
                    nn.predecessors = []
                    
                    for pred in range(0, nn.numlayer+1):
                        predecessor = np.zeros(1, dtype=np.int)
                        predecessor[0] = int(pred-1)
                        nn.predecessors.append(predecessor)
                    #print("predecessors ", nn.predecessors[0][0])
                    for labels in range(num_outputs):
                        #print("num_outputs ", num_outputs, nn.numlayer, len(nn.weights[-1]))
                        if labels != int(test[0]):
                            if res[var][0] < 0:
                                labels_to_be_verified.append(labels)
                            var = var+1
                    #print("relu layers", relu_layers)

                    is_verified, x = refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, int(test[0]),
                                                            labels_to_be_verified, K=config.k, s=config.s,
                                                            complete=config.complete,
                                                            timeout_final_lp=config.timeout_final_lp,
                                                            timeout_final_milp=config.timeout_final_milp,
                                                            timeout_lp=config.timeout_lp,
                                                            timeout_milp=config.timeout_milp,
                                                            use_milp=config.use_milp,
                                                            partial_milp=config.partial_milp,
                                                            max_milp_neurons=config.max_milp_neurons,
                                                            approx=config.approx_k)
                    if is_verified:
                        print("img", i, "Verified", int(test[0]))
                        verified_images += 1
                    else:
                        if x != None:
                            adv_image = np.array(x)
                            res = np.argmax((network.eval(adv_image))[:,0])
                            if res!=int(test[0]):
                                denormalize(x,means, stds, dataset)
                                # print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", int(test[0]))
                                print("img", i, "Verified unsafe against label ", res, "correct label ", int(test[0]))
                                unsafe_images += 1

                            else:
                                print("img", i, "Failed")
                        else:
                            print("img", i, "Failed")
                else:
                    print("img", i, "Failed")
            else:
                if domain.endswith("poly"):
                    perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop, K=0, s=0,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=False,
                                                                                      complete=False,
                                                                                      terminate_on_failure=not config.complete,
                                                                                      partial_milp=0,
                                                                                      max_milp_neurons=0,
                                                                                      approx_k=0)
                    print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
                if not domain.endswith("poly") or not (perturbed_label==label):
                    perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, domain,
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop,
                                                                                      K=config.k, s=config.s,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=config.use_milp,
                                                                                      complete=config.complete,
                                                                                      terminate_on_failure=not config.complete,
                                                                                      partial_milp=config.partial_milp,
                                                                                      max_milp_neurons=config.max_milp_neurons,
                                                                                      approx_k=config.approx_k)
                    print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_labels)
                if (perturbed_label==label):
                    print("img", i, "Verified", label)
                    verified_images += 1
                else:
                    if complete==True and failed_labels is not None:
                        failed_labels = list(set(failed_labels))
                        constraints = get_constraints_for_dominant_label(label, failed_labels)
                        verified_flag, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                        if(verified_flag==True):
                            print("img", i, "Verified as Safe using MILP", label)
                            verified_images += 1
                        else:
                            if adv_image != None:
                                cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                                if(cex_label!=label):
                                    denormalize(adv_image[0], means, stds, dataset)
                                    # print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                                    print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                                    unsafe_images+=1
                                else:
                                    print("img", i, "Failed with MILP, without a adeversarial example")
                            else:
                                print("img", i, "Failed with MILP")
                    else:
                    
                        if x != None:
                            cex_label,_,_,_,_,_ = eran.analyze_box(x,x,'deepzono',config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                            print("cex label ", cex_label, "label ", label)
                            if(cex_label!=label):
                                denormalize(x,means, stds, dataset)
                                # print("img", i, "Verified unsafe with adversarial image ", x, "cex label ", cex_label, "correct label ", label)
                                print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                                unsafe_images += 1
                            else:
                                print("img", i, "Failed, without a adversarial example")
                        else:
                            print("img", i, "Failed")

            end = time.time()
            cum_time += end - start # only count samples where we did try to certify
        else:
            print("img",i,"not considered, incorrectly classified")
            end = time.time()

        print(f"progress: {1 + i - config.from_test}/{config.num_tests}, "
              f"correct:  {correctly_classified_images}/{1 + i - config.from_test}, "
              f"verified: {verified_images}/{correctly_classified_images}, "
              f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
              f"time: {end - start:.3f}; {0 if cum_time==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")



    print('analysis precision ',verified_images,'/ ', correctly_classified_images)
