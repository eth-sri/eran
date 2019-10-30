import sys
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(1, '../tf_verify/')
import numpy as np
import os
from eran import ERAN
from read_net_file import *
import tensorflow as tf
import csv
from deepzono_milp import *
import onnxruntime.backend as rt
import argparse
from onnx import helper


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

def normalize(image, means, stds):
    if(dataset=='mnist'):
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

def denormalize(image, means, stds):
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
        assert args.dataset, "if you define specific network(s), you must difine their dataset."
        networks = args.network
        dataset_folder = ''
    else:
        dataset_folder = '../data/test_nets/' + dataset + '/'
        networks = os.listdir(dataset_folder)

    for network in networks:
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
                tested_file.write(', '.join([dataset, network, 'ERAN parse error message: ' + str(e)]) + '\n')
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
                # this is a hack and should be done nicer
                if dataset == 'cifar10':
                    means=[0.485, 0.456, 0.406]
                    stds=[0.225, 0.225, 0.225]
                else:
                    means = [0]
                    stds = [1]
            else:
                sess = tf.Session()
                model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
            try:
                eran = ERAN(model, is_onnx=is_onnx)
            except Exception as e:
                tested_file.write(', '.join([dataset, network, 'ERAN parse error message: ' + str(e)]) + '\n')
                tested_file.flush()
                continue

        if args.parser_only:
            tested_file.write(', '.join([dataset, network, 'ERAN parsed successfully\n']))
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


            if(dataset=='mnist'):
                image= np.float64(test[1:len(test)])/np.float64(255)
            else:
                if is_trained_with_pytorch or is_onnx:
                    image= (np.float64(test[1:len(test)])/np.float64(255))
                else:
                    image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5

            specLB = np.copy(image)
            specUB = np.copy(image)
            test_input = np.copy(image)

            if is_trained_with_pytorch or is_onnx:
                normalize(specLB, means, stds)
                normalize(specUB, means, stds)
                normalize(test_input, means, stds)

            print(', '.join([dataset, network, domain]), 'testing now')

            try:
                label, nn, nlb, nub, output_info = eran.analyze_box(specLB, specUB, domain, 1, 1, True, testing=True)

            except Exception as e:
                tested_file.write(', '.join([dataset, network, domain, 'ERAN analyze error message: ' + str(e)]) + '\n')
                tested_file.flush()
                continue

            if dataset == 'cifar10':
                input = np.array(test_input, dtype=np.float32).reshape([1, 32, 32, 3])
            elif dataset == 'mnist':
                input = np.array(test_input, dtype=np.float32).reshape([1, 28, 28, 1])

            if is_onnx:
                input = input.transpose(0, 3, 1, 2)

            if is_onnx:
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
                pred = pred[-1]
            else:
                if not (is_saved_tf_model or is_pb_file):
                    input = np.array(test_input, dtype=np.float32)
                output_names = [e[0] for e in output_info]
                pred = sess.run(get_out_tensors(output_names), {sess.graph.get_operations()[0].name + ':0': input})
                #print(pred)
                pred = pred[-1]
            pred_eran = np.asarray([(i+j)/2 for i, j in zip(nlb[-1], nub[-1])])
            pred = np.asarray(pred).reshape(-1)
            if len(pred_eran) != len(pred):
                tested_file.write(', '.join([dataset, network, domain, 'predictions have not the same number of labels. ERAN: ' + len(pred_eran) + ' model: ' + len(pred)]) + '\n')
                tested_file.flush()
                continue
            difference = pred_eran - pred
            if np.all([abs(elem) < .001 for elem in difference]):
                tested_file.write(', '.join([dataset, network, domain, 'success']) + '\n')
                tested_file.flush()
            else:
                i = 0
                if is_onnx:
                    i = 1

                tested_file.write(', '.join([dataset, network, domain, str(pred_eran), str(pred)]) + '\n')
                tested_file.flush()