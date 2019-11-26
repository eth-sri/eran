import sys
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../geometric/code/')
import numpy as np
import os
from eran import ERAN
from geometric_constraints import *
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_acasxu_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    low = []
    high = []
    for line in text.split('\n'):
        if line!="":
           [lb,ub] = line.split(",")
           low.append(np.double(lb))
           high.append(np.double(ub))
    return low,high


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


def normalize(image, means, stds, dataset, is_conv):
    if(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        tmp = np.zeros(3072)
        for i in range(3072):
            tmp[i] = (image[i] - means[i % 3]) / stds[i % 3]

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(3072):
                image[i] = (tmp[i % 1024 + (i % 3) * 1024] - means[i % 3]) / stds[i % 3]


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
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
    if(dataset=='mnist'):
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


def get_tests(dataset):
    if (dataset == 'acasxu'):
        specfile = '../data/acasxu/specs/acasxu_prop' + str(specnumber) + '_spec.txt'
        tests = open(specfile, 'r').read()
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
parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .pyt, .tf and .meta')
parser.add_argument('--epsilon', type=float, default=0, help='the epsilon for L_infinity perturbation')
parser.add_argument('--zonotope', type=str, default=None, help='file to specify the zonotope matrix')
#parser.add_argument('--specnumber', type=int, default=9, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=None, help='the domain name can be either deepzono, refinezono, deeppoly or refinepoly')
parser.add_argument('--dataset', type=str, default=None, help='the dataset, can be either mnist, cifar10, or acasxu')
parser.add_argument('--complete', type=str2bool, default=False,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=1,  help='timeout for the LP solver')
parser.add_argument('--timeout_milp', type=float, default=1,  help='timeout for the MILP solver')
parser.add_argument('--use_area_heuristic', type=str2bool, default=True,  help='whether to use area heuristic for the DeepPoly ReLU approximation')
parser.add_argument('--mean', nargs='+', type=float,  help='the mean used to normalize the data with')
parser.add_argument('--std', nargs='+', type=float,  help='the standard deviation used to normalize the data with')
parser.add_argument('--config', type=str, help='config location')
parser.add_argument('--num_params', type=int, default=0, help='Number of transformation parameters')
parser.add_argument('--num_tests', type=int, default=None, help='Number of images to test')
parser.add_argument('--from_test', type=int, default=0, help='Number of images to test')
parser.add_argument('--test_idx', type=int, default=None, help='Index to test')
parser.add_argument('--debug', action='store_true', help='Whether to display debug info')
parser.add_argument('--attack', action='store_true', help='Whether to attack')
parser.add_argument('--geometric', '-g', dest='geometric', action='store_true', help='Whether to attack')

args = parser.parse_args()

#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

netname = args.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = args.epsilon
assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = args.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = args.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not args.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain name can be either deepzono, refinezono, deeppoly or refinepoly"

dataset = args.dataset

if zonotope_bool==False:
   assert dataset in ['mnist','cifar10','acasxu'], "only mnist, cifar10, and acasxu datasets are supported"


specnumber = 9
if(dataset=='acasxu' and (specnumber!=9)):
    print("currently we only support property 9 for acasxu")
    exit(1)


is_conv = False
mean = 0
std = 0

complete = (args.complete==True)

if(dataset=='acasxu'):
    print("netname ", netname, " specnumber ", specnumber, " domain ", domain, " dataset ", dataset, "args complete ", args.complete, " complete ",complete, " timeout_lp ",args.timeout_lp)
else:
    print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", args.complete, " complete ",complete, " timeout_lp ",args.timeout_lp)

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
        is_trained_with_pytorch = True
        model, is_conv = read_onnx_net(netname)
        # this is to have different defaults for mnist and cifar10
        if dataset == 'cifar10':
            means=[0.485, 0.456, 0.406]
            stds=[0.225, 0.225, 0.225]
        else:
            means = [0]
            stds = [1]
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
    eran = ERAN(model, is_onnx=is_onnx)

if args.mean:
    means = args.mean
if args.std:
    stds = args.std

correctly_classified_images = 0
verified_images = 0
total_images = 0


if dataset:
    tests = get_tests(dataset)


if dataset=='acasxu':
    # Ignores Zonotope for now.
    specLB, specUB = parse_acasxu_spec(tests)
    if(specnumber==9):
        num_splits = [10,9,1,5,14]
    else:
        num_splits = [10,10,1,10,10]
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
    _,nn,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), args.timeout_lp, args.timeout_milp, args.use_area_heuristic, specnumber)
    start = time.time()
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

                        label,_,nlb,nub = eran.analyze_box(specLB, specUB, domain, args.timeout_lp, args.timeout_milp, args.use_area_heuristic, specnumber)

                        if(specnumber==9 and label!=3):
                            if complete==True:
                               verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, 3, nlb, nub,False)
                               if(verified_flag==False):
                                  flag = False
                                  break
                            else:
                               flag = False
                               break
    end = time.time()
    if(flag):
        print("acasxu property ", specnumber, "Verified")
    else:
        print("acasxu property ", specnumber, "Failed")

    print(end - start, "seconds")

elif zonotope_bool:
    perturbed_label, nn, nlb, nub = eran.analyze_zonotope(zonotope, domain, args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
    print("nlb ",nlb[len(nlb)-1])
    #print("nub ",nub)
    if(perturbed_label!=-1):
         print("Verified")
    #elif(complete==True):
    #     verified_flag,adv_image = verify_network_with_milp_zonotope(nn, zonotope, label, nlb, nub)
    #     if(verified_flag==True):
    #         print("Verified")
    #     else:
    #         print("Failed")
    else:
         print("Failed")


elif args.geometric:
    total, attacked, standard_correct, tot_time = 0, 0, 0, 0
    correct_box, correct_poly = 0, 0
    cver_box, cver_poly = [], []

    transform_attack_container = get_transform_attack_container(args.config)
    for i, test in enumerate(tests):
        set_transform_attack_for(transform_attack_container, i)
        if args.test_idx is not None and i != args.test_idx:
            continue
        attack_params = get_attack_params(transform_attack_container)
        attack_images = get_attack_images(transform_attack_container)
        if args.num_tests is not None and i >= args.num_tests:
            break
        print('Test {}:'.format(i))

        if args.dataset == 'mnist' or args.dataset == 'fashion':
            image = np.float64(test[1:len(test)])
            n_rows, n_cols, n_channels = 28, 28, 1
        else:
            n_rows, n_cols, n_channels = 32, 32, 3
            if is_trained_with_pytorch:
                image = np.float64(test[1:len(test)])
            else:
                image = np.float64(test[1:len(test)]) - 0.5

        spec_lb = np.copy(image)
        spec_ub = np.copy(image)

        if (is_trained_with_pytorch):
            normalize(spec_lb, means, stds, args.dataset, is_conv)
            normalize(spec_ub, means, stds, args.dataset, is_conv)

        label, nn, nlb, nub = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', args.timeout_lp, args.timeout_milp,
                                               args.use_area_heuristic)
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
        k = args.num_params + 1 + 1 + dim

        attack_imgs, checked, attack_pass = [], [], 0
        cex_found = False
        if args.attack:
            for j in tqdm(range(0, len(attack_params))):
                params = attack_params[j]
                values = np.array(attack_images[j])

                attack_lb = values[::2]
                attack_ub = values[1::2]

                if is_trained_with_pytorch:
                    normalize(attack_lb, means, stds, args.dataset, is_conv)
                    normalize(attack_ub, means, stds, args.dataset, is_conv)
                else:
                    attack_lb -= 0.5
                    attack_ub -= 0.5
                attack_imgs.append((params, attack_lb, attack_ub))
                checked.append(False)

                predict_label, _, _, _ = eran.analyze_box(
                    attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                    args.timeout_lp, args.timeout_milp, args.use_area_heuristic, 0)
                if predict_label != int(test[0]):
                    print('counter-example, params: ', params, ', predicted label: ', predict_label)
                    cex_found = True
                    break
                else:
                    attack_pass += 1
        print('tot attacks: ', len(attack_imgs))

        lines = get_transformations(transform_attack_container)
        lines.append([])
        print('Number of lines: ', len(lines))
        assert len(lines) % k == 0

        spec_lb = np.zeros(args.num_params + dim)
        spec_ub = np.zeros(args.num_params + dim)

        expr_size = args.num_params
        lexpr_cst, uexpr_cst = [], []
        lexpr_weights, uexpr_weights = [], []
        lexpr_dim, uexpr_dim = [], []

        ver_chunks_box, ver_chunks_poly, tot_chunks = 0, 0, 0

        for i, line in enumerate(lines):
            if i % k < args.num_params:
                # read specs for the parameters
                values = line
                assert len(values) == 2
                param_idx = i % k
                spec_lb[dim + param_idx] = values[0]
                spec_ub[dim + param_idx] = values[1]
                if args.debug:
                    print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
            elif i % k == args.num_params:
                # read interval bounds for image pixels
                values = line
                spec_lb[:dim] = values[::2]
                spec_ub[:dim] = values[1::2]
                # if args.debug:
                #     show_ascii_spec(spec_lb, spec_ub)
            elif i % k < k - 1:
                # read polyhedra constraints for image pixels
                tokens = line
                assert len(tokens) == 2 + 2 * args.num_params

                bias_lower, weights_lower = tokens[0], tokens[1:1 + args.num_params]
                bias_upper, weights_upper = tokens[args.num_params + 1], tokens[2 + args.num_params:]

                assert len(weights_lower) == args.num_params
                assert len(weights_upper) == args.num_params

                lexpr_cst.append(bias_lower)
                uexpr_cst.append(bias_upper)
                for j in range(args.num_params):
                    lexpr_dim.append(dim + j)
                    uexpr_dim.append(dim + j)
                    lexpr_weights.append(weights_lower[j])
                    uexpr_weights.append(weights_upper[j])
            else:
                assert (len(line) == 0)
                for p_idx in range(args.num_params):
                    lexpr_cst.append(spec_lb[dim + p_idx])
                    for l in range(args.num_params):
                        lexpr_weights.append(0)
                        lexpr_dim.append(dim + l)
                    uexpr_cst.append(spec_ub[dim + p_idx])
                    for l in range(args.num_params):
                        uexpr_weights.append(0)
                        uexpr_dim.append(dim + l)
                if (is_trained_with_pytorch):
                    normalize(spec_lb[:dim], means, stds, args.dataset, is_conv)
                    normalize(spec_ub[:dim], means, stds, args.dataset, is_conv)
                normalize_poly(args.num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights,
                               uexpr_dim, means, stds, args.dataset)

                for attack_idx, (attack_params, attack_lb, attack_ub) in enumerate(attack_imgs):
                    ok_attack = True
                    for j in range(num_pixels):
                        low, up = lexpr_cst[j], uexpr_cst[j]
                        for idx in range(args.num_params):
                            low += lexpr_weights[j * args.num_params + idx] * attack_params[idx]
                            up += uexpr_weights[j * args.num_params + idx] * attack_params[idx]
                        if low > attack_lb[j] + EPS or attack_ub[j] > up + EPS:
                            ok_attack = False
                    if ok_attack:
                        checked[attack_idx] = True
                        # print('checked ', attack_idx)
                if args.debug:
                    print('Running the analysis...')

                t_begin = time.time()
                perturbed_label_poly, _, _, _ = eran.analyze_box(
                    spec_lb, spec_ub, 'deeppoly',
                    args.timeout_lp, args.timeout_milp, args.use_area_heuristic, 0,
                    lexpr_weights, lexpr_cst, lexpr_dim,
                    uexpr_weights, uexpr_cst, uexpr_dim,
                    expr_size)
                perturbed_label_box, _, _, _ = eran.analyze_box(
                    spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                    args.timeout_lp, args.timeout_milp, args.use_area_heuristic, 0)
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
        if(dataset=='mnist'):
            image= np.float64(test[1:len(test)])/np.float64(255)
        else:
            if is_trained_with_pytorch:
                image= (np.float64(test[1:len(test)])/np.float64(255))
            else:
                image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5

        specLB = np.copy(image)
        specUB = np.copy(image)

        if is_trained_with_pytorch:
            normalize(specLB, means, stds, dataset, is_conv)
            normalize(specUB, means, stds, dataset, is_conv)

        label,nn,nlb,nub = eran.analyze_box(specLB, specUB, init_domain(domain), args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
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

            if(dataset=='mnist'):
                specLB = np.clip(image - epsilon,0,1)
                specUB = np.clip(image + epsilon,0,1)
            else:
                if(is_trained_with_pytorch):
                     specLB = np.clip(image - epsilon,0,1)
                     specUB = np.clip(image + epsilon,0,1)
                else:
                     specLB = np.clip(image-epsilon,-0.5,0.5)
                     specUB = np.clip(image+epsilon,-0.5,0.5)
            if(is_trained_with_pytorch):
                normalize(specLB, means, stds, dataset, is_conv)
                normalize(specUB, means, stds, dataset, is_conv)
            start = time.time()
            perturbed_label, _, nlb, nub = eran.analyze_box(specLB, specUB, domain, args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
            print("nlb ", nlb[len(nlb)-1], " nub ", nub[len(nub)-1])
            if(perturbed_label==label):
                print("img", total_images, "Verified", label)
                verified_images += 1
            else:
                if complete==True:
                    verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, label, nlb, nub)
                    if(verified_flag==True):
                        print("img", total_images, "Verified", label)
                        verified_images += 1
                    else:
                        print("img", total_images, "Failed")
                        cex_label,_,_,_ = eran.analyze_box(adv_image, adv_image, 'deepzono', args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
                        if(cex_label!=label):
                            if(is_trained_with_pytorch):
                                denormalize(adv_image, means, stds, dataset)
                            print("adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                else:
                    print("img", total_images, "Failed")

            correctly_classified_images +=1
            end = time.time()
            print(end - start, "seconds")
        else:
            print("img",total_images,"not considered, correct_label", int(test[0]), "classified label ", label)
        total_images += 1

    print('analysis precision ',verified_images,'/ ', correctly_classified_images)