import sys
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(1, '../tf_verify/')
import traceback
import numpy as np
import os
from eran import ERAN
from read_net_file import *
import tensorflow as tf
import csv
from ai_milp import *
import onnxruntime.backend as rt
import argparse
from onnx import helper
from config import config

parser = argparse.ArgumentParser(description='ERAN sanity check',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str,  help='the dataset to test with')
parser.add_argument('--domain', nargs='+', type=str, default=['deepzono', 'refinezono', 'deeppoly', 'refinepoly'],  help='the domains to be tested. default:deepzono refinezono deeppoly refinepoly')
parser.add_argument('--network', nargs='+', type=str,  help='the networks to be tested (if this option is taken --dataset MUST be given.')
parser.add_argument('--out', '-o', default='tested.txt', type=str,  help='the filename where the result will be saved. default: tested.txt')
parser.add_argument('--parser', '-p', dest='parser_only', action='store_true',  help='with this flag only the parser will be tested')
parser.add_argument('--continue', '-c', dest='new_only', action='store_true',  help='with this flag only networks without a previous test result will be tested')
parser.add_argument('--failed', '-f', dest='failed_only', action='store_true',  help='with this flag only networks that have not passed a previous test will be tested')
parser.set_defaults(parser_only=False)
parser.set_defaults(new_only=False)
parser.set_defaults(failed_only=False)
args = parser.parse_args()


def normalize(image):
    for i in range(len(image)):
        image[i] -= config.mean[i]
        image[i] /= config.std[i]



domains = args.domain
non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']
result_file = args.out
tested_file = open(result_file, 'a+')
already_tested = open(result_file, 'r').read()

def get_out_tensors(out_names):
    return [sess.graph.get_tensor_by_name(name[7:]) for name in out_names]

if args.dataset:
    datasets = [args.dataset]
else:
    datasets = os.listdir('../data/test_nets/')

for dataset in datasets:
    if args.network:
        assert args.dataset, "if you define specific network(s), you must define their dataset."
        networks = args.network
        dataset_folder = ''
    else:
        dataset_folder = '../data/test_nets/' + dataset + '/'
        networks = os.listdir(dataset_folder)

    for network in networks:
        # reset mean and std for new network
        config.mean = None
        config.std = None
        if args.new_only:
            tested_for_all_domains = True
            for domain in domains:
                if not ', '.join([dataset, network, domain]) in already_tested:
                    tested_for_all_domains = False
                    break
            if tested_for_all_domains:
                continue
        if args.failed_only:
            success_for_all_domains = True
            for domain in domains:
                if not ', '.join([dataset, network, domain, 'success']) in already_tested:
                    success_for_all_domains = False
                    break
            if success_for_all_domains:
                continue
        tf.reset_default_graph()
        netname = dataset_folder + network
        filename, file_extension = os.path.splitext(netname)

        is_trained_with_pytorch = file_extension==".pyt"
        is_saved_tf_model = file_extension==".meta"
        is_pb_file = file_extension==".pb"
        is_tensorflow = file_extension== ".tf"
        is_onnx = file_extension == ".onnx"

        complete = False

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
            out_tensor = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')
            try:
                eran = ERAN(out_tensor, sess)
            except Exception as e:
                tested_file.write(', '.join([dataset, network, 'ERAN parse error trace: '+traceback.format_exc()]) + '\n\n\n')
                tested_file.flush()
                continue

        else:
            if(dataset=='mnist'):
                num_pixels = 784
            elif (dataset=='cifar10'):
                num_pixels = 3072
            elif(dataset=='acasxu'):
                num_pixels = 5
            if is_onnx:
                model, is_conv = read_onnx_net(netname)
            else:
                sess = tf.Session()
                model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
            try:
                eran = ERAN(model, is_onnx=is_onnx)
            except Exception as e:
                tested_file.write(', '.join([dataset, network, 'ERAN parse error trace: ' + traceback.format_exc()]) + '\n\n\n')
                tested_file.flush()
                continue

        if args.parser_only:
            tested_file.write(', '.join([dataset, network, 'ERAN parsed successfully\n', str(eran.optimizer.operations), '\n']))
            tested_file.flush()
            continue

        if(dataset=='mnist'):
            csvfile = open('../data/mnist_test.csv', 'r')
            tests = csv.reader(csvfile, delimiter=',')
        elif(dataset=='cifar10'):
            csvfile = open('../data/cifar10_test.csv', 'r')
            tests = csv.reader(csvfile, delimiter=',')
        else:
            specfile = '../data/acasxu/specs/acasxu_prop' + str(specnumber) +'_spec.txt'
            tests = open(specfile, 'r').read()

        test = next(tests)


        for domain in domains:

            if args.new_only and ', '.join([dataset, network, domain]) in already_tested:
                print(', '.join([dataset, network, domain]), 'already tested.')
                continue

            if args.failed_only and ', '.join([dataset, network, domain, 'success']) in already_tested:
                print(', '.join([dataset, network, domain]), 'already successfully tested.')
                continue


            image= np.float64(test[1:len(test)])/np.float64(255)

            specLB = np.copy(image)
            specUB = np.copy(image)
            test_input = np.copy(image)

            if config.mean is not None:
                normalize(specLB)
                normalize(specUB)

            print(', '.join([dataset, network, domain]), 'testing now')

            try:
                label, nn, nlb, nub, output_info = eran.analyze_box(specLB, specUB, domain, 1, 1, config.use_area_heuristic, testing=True)

            except Exception as e:
                tested_file.write(', '.join([dataset, network, domain, 'ERAN analyze error trace: '+traceback.format_exc()]) + '\n\n\n')
                tested_file.flush()
                continue

            if dataset == 'cifar10':
                input = np.array(test_input, dtype=np.float32).reshape([1, 32, 32, 3])
            elif dataset == 'mnist':
                input = np.array(test_input, dtype=np.float32).reshape([1, 28, 28, 1])

            if is_onnx:
                input = input.transpose(0, 3, 1, 2)
                for name, shape in output_info:
                    out_node = helper.ValueInfoProto(type = helper.TypeProto())
                    out_node.name = name
                    out_node.type.tensor_type.elem_type = model.graph.output[0].type.tensor_type.elem_type
                    if len(shape)==4:
                        shape = [shape[0], shape[3], shape[1], shape[2]]
                    for dim_value in shape:
                        dim = out_node.type.tensor_type.shape.dim.add()
                        dim.dim_value = dim_value
                    model.graph.output.append(out_node)
                runnable = rt.prepare(model, 'CPU')
                pred = runnable.run(input)
                #print(pred)
            else:
                if not (is_saved_tf_model or is_pb_file):
                    input = np.array(test_input, dtype=np.float32)
                output_names = [e[0] for e in output_info]
                pred = sess.run(get_out_tensors(output_names), {sess.graph.get_operations()[0].name + ':0': input})
                #print(pred)
            pred_eran = np.asarray([(i+j)/2 for i, j in zip(nlb[-1], nub[-1])])
            pred_model = np.asarray(pred[-1]).reshape(-1)
            if len(pred_eran) != len(pred_model):
                tested_file.write(', '.join([dataset, network, domain, 'predictions have not the same number of labels. ERAN: ' + str(len(pred_eran)) + ' model: ' + str(len(pred_model))]) + '\n\n\n')
                tested_file.flush()
                continue
            difference = pred_eran - pred_model
            if np.all([abs(elem) < .001 for elem in difference]):
                tested_file.write(', '.join([dataset, network, domain, 'success']) + '\n')
                tested_file.flush()
            else:
                tested_file.write(', '.join([dataset, network, domain, '\neran', str(pred_eran), '\nmodel', str(pred_model)]) + '\n')
                tested_file.flush()
                for i in range(len(nlb)):
                    pred_eran = np.asarray([(l+u)/2 for l, u in zip(nlb[i], nub[i])])
                    offset = len(pred) - len(nlb)
                    pred_model = np.asarray(pred[i + offset])
                    if is_onnx and pred_model.ndim == 4:
                        pred_model = pred_model.transpose(0, 2, 3, 1)
                    difference = pred_eran - pred_model.reshape(-1)
                    if not np.all([abs(elem) < .001 for elem in difference]):
                        tested_file.write(', '.join([dataset, network, domain, 'started divergence at layer', str(i), 'outputname', str(output_info[i+1][0]), '\ndifference', str(difference)]) + '\n\n\n')
                        tested_file.flush()
                        break