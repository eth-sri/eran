import sys
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import numpy as np
import os
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
from constraints import *
import re
import itertools

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

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
    return boxes


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

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1
        


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


def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
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
parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly or refinepoly')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
parser.add_argument('--dyn_krelu', action='store_true', default=config.dyn_krelu, help='dynamically select parameter k')
parser.add_argument('--use_2relu', action='store_true', default=config.use_2relu, help='use 2-relu')
parser.add_argument('--use_3relu', action='store_true', default=config.use_3relu, help='use 3-relu')
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


# Logging options
parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')


args = parser.parse_args()
for k, v in vars(args).items():
    setattr(config, k, v)
config.json = vars(args)

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
assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = config.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = config.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not config.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain name can be either deepzono, refinezono, deeppoly or refinepoly"

dataset = config.dataset

if zonotope_bool==False:
   assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

constraints = None
if config.output_constraints:
    constraints = get_constraints_from_file(config.output_constraints)

mean = 0
std = 0
is_conv = False

complete = (config.complete==True)

if(dataset=='acasxu'):
    print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
else:
    print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

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
    eran = ERAN(sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0'), sess)

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
        # this is to have different defaults for mnist and cifar10
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
    eran = ERAN(model, is_onnx=is_onnx)

if not is_trained_with_pytorch:
    if dataset == 'mnist' and not config.geometric:
        means = [0]
        stds = [1]
    elif dataset == 'acasxu':
        means = [1.9791091e+04,0.0,0.0,650.0,600.0]
        stds = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]

is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std

correctly_classified_images = 0
verified_images = 0


if dataset:
    if config.input_box is None:
        tests = get_tests(dataset, config.geometric)
    else:
        tests = open(config.input_box, 'r').read()


if dataset=='acasxu':
    if config.debug:
        print('Constraints: ', constraints)
    boxes = parse_input_box(tests)
    for box in boxes:
        specLB = [interval[0] for interval in box]
        specUB = [interval[1] for interval in box]
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)
        if config.specnumber == 9:
            num_splits = [10,9,1,5,14]
        elif config.specnumber == 5:
            num_splits = [4,5,1,20,20]
        else:
            num_splits = [10, 10, 1, 10, 10]
        step_size = []
        for i in range(5):
            step_size.append((specUB[i]-specLB[i])/num_splits[i])
        #sorted_indices = np.argsort(widths)
        #input_to_split = sorted_indices[0]
        #print("input to split ", input_to_split)

        #step_size = widths/num_splits
        start_val = np.copy(specLB)
        end_val = np.copy(specUB)
        flag = True
        _,nn,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
        start = time.time()
        #complete_list = []
        for i in range(num_splits[0]):
            specLB[0] = start_val[0] + i*step_size[0]
            specUB[0] = np.fmin(end_val[0],start_val[0]+ (i+1)*step_size[0])

            for j in range(num_splits[1]):
                specLB[1] = start_val[1] + j*step_size[1]
                specUB[1] = np.fmin(end_val[1],start_val[1]+ (j+1)*step_size[1])

                for k in range(num_splits[2]):
                    specLB[2] = start_val[2] + k*step_size[2]
                    specUB[2] = np.fmin(end_val[2],start_val[2]+ (k+1)*step_size[2])
                    for l in range(num_splits[3]):
                        specLB[3] = start_val[3] + l*step_size[3]
                        specUB[3] = np.fmin(end_val[3],start_val[3]+ (l+1)*step_size[3])
                        for m in range(num_splits[4]):

                            specLB[4] = start_val[4] + m*step_size[4]
                            specUB[4] = np.fmin(end_val[4],start_val[4]+ (m+1)*step_size[4])

                            hold,_,nlb,nub = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)

                            if not hold:
                                if complete==True:
                                   verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                                   #complete_list.append((i,j,k,l,m))
                                   if(verified_flag==False):
                                      flag = False
                                      assert 0
                                else:
                                   flag = False
                                   break
                            if config.debug:
                                print('split', i, j, k, l, m)
        end = time.time()
        #print(complete_list)
        if(flag):
            print("acasxu property ", config.specnumber, "Verified")
        else:
            print("acasxu property ", config.specnumber, "Failed")

        print(end - start, "seconds")

elif zonotope_bool:
    perturbed_label, nn, nlb, nub = eran.analyze_zonotope(zonotope, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
    print("nlb ",nlb[-1])
    print("nub ",nub[-1])
    if(perturbed_label!=-1):
        print("Verified")
    elif(complete==True):
        constraints = get_constraints_for_dominant_label(perturbed_label, 10)
        verified_flag,adv_image = verify_network_with_milp(nn, zonotope, [], nlb, nub, constraints)
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

            label, nn, nlb, nub = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', config.timeout_lp, config.timeout_milp,
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

                    predict_label, _, _, _ = eran.analyze_box(
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
                    perturbed_label_poly, _, _, _ = eran.analyze_box(
                        spec_lb, spec_ub, 'deeppoly',
                        config.timeout_lp, config.timeout_milp, config.use_default_heuristic, None,
                        lexpr_weights, lexpr_cst, lexpr_dim,
                        uexpr_weights, uexpr_cst, uexpr_dim,
                        expr_size)
                    perturbed_label_box, _, _, _ = eran.analyze_box(
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

            label, nn, nlb, nub = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', config.timeout_lp, config.timeout_milp,
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

                        predict_label, _, _, _ = eran.analyze_box(
                            attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                            config.timeout_lp, config.timeout_milp, config.use_default_heuristic, 0)
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
                        perturbed_label_poly, _, _, _ = eran.analyze_box(
                            spec_lb, spec_ub, 'deeppoly',
                            config.timeout_lp, config.timeout_milp, config.use_default_heuristic, 0,
                            lexpr_weights, lexpr_cst, lexpr_dim,
                            uexpr_weights, uexpr_cst, uexpr_dim,
                            expr_size)
                        perturbed_label_box, _, _, _ = eran.analyze_box(
                            spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                            config.timeout_lp, config.timeout_milp, config.use_default_heuristic, 0)
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
else:
    for i, test in enumerate(tests):
        if config.from_test and i < config.from_test:
            continue

        if config.num_tests is not None and i >= config.num_tests:
            break

        image= np.float64(test[1:len(test)])/np.float64(255)

        specLB = np.copy(image)
        specUB = np.copy(image)

        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)

        label,nn,nlb,nub = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
        #for number in range(len(nub)):
        #    for element in range(len(nub[number])):
        #        if(nub[number][element]<=0):
        #            print('False')
        #        else:
        #            print('True')

        print("concrete ", nlb[-1])
        #if(label == int(test[0])):
        if(label == int(test[0])):
            perturbed_label = None

            specLB = np.clip(image - epsilon,0,1)
            specUB = np.clip(image + epsilon,0,1)
            normalize(specLB, means, stds, dataset)
            normalize(specUB, means, stds, dataset)
            start = time.time()
            perturbed_label, _, nlb, nub = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
            print("nlb ", nlb[len(nlb)-1], " nub ", nub[len(nub)-1])
            if(perturbed_label==label):
                print("img", i, "Verified", label)
                verified_images += 1
            else:
                if complete==True:
                    constraints = get_constraints_for_dominant_label(label, 10)
                    verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                    if(verified_flag==True):
                        print("img", i, "Verified", label)
                        verified_images += 1
                    else:
                        print("img", i, "Failed")
                        cex_label,_,_,_ = eran.analyze_box(adv_image, adv_image, 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                        if(cex_label!=label):
                            denormalize(adv_image, means, stds, dataset)
                            print("adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                else:
                    print("img", i, "Failed")

            correctly_classified_images +=1
            end = time.time()
            print(end - start, "seconds")
        else:
            print("img",i,"not considered, correct_label", int(test[0]), "classified label ", label)

    print('analysis precision ',verified_images,'/ ', correctly_classified_images)
