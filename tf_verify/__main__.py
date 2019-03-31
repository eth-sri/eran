import sys
sys.path.insert(0, '../ELINA/python_interface/')
import numpy as np
import os
from eran import ERAN
from read_net_file import *
import tensorflow as tf
import csv
import time
from deepzono_milp import * 
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .pyt, .tf and .meta')
parser.add_argument('--epsilon', type=float, default=0, help='the epsilon for L_infinity perturbation')
parser.add_argument('--domain', type=str, default=None, help='the domain name can be either deepzono, refinezono or deeppoly')
parser.add_argument('--dataset', type=str, default=None, help='the dataset, can be either mnist or cifar10')
parser.add_argument('--complete', type=str2bool, default=False,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=1,  help='timeout for the LP solver')
parser.add_argument('--timeout_milp', type=float, default=1,  help='timeout for the MILP solver')

args = parser.parse_args()

#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

netname = args.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = False
is_saved_tf_model = False

if(file_extension==".pyt"):
    is_trained_with_pytorch = True
elif(file_extension==".meta"):
    is_saved_tf_model = True
elif(file_extension!= ".tf"):
    print("file extension not supported")
    exit(1)


epsilon = args.epsilon
if((epsilon<0) or (epsilon>1)):
    print("epsilon can only be between 0 and 1")
    exit(1)

domain = args.domain

if((domain!='deepzono') and (domain!='refinezono') and (domain!='deeppoly')):
    print("domain name can be either deepzono, refinezono or deeppoly")
    exit(1)

dataset = args.dataset
if((dataset!='mnist') and (dataset!='cifar10')):
    print("only mnist and cifar10 dataset are supported")
    exit(1)




is_conv = False
mean = 0
std = 0

complete = (domain=='deepzono') and (args.complete==True)

print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", args.complete, " complete ",complete, " timeout_lp ",args.timeout_lp)
if(is_saved_tf_model):
    netfolder = os.path.dirname(netname) 

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    saver = tf.train.import_meta_graph(netname)
    saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    eran = ERAN(sess.graph.get_tensor_by_name('logits:0'), sess)    

else:
    if(dataset=='mnist'):
        num_pixels = 784
    else:
        num_pixels = 3072

    model, is_conv, means, stds = read_net(netname, num_pixels, is_trained_with_pytorch)
    eran = ERAN(model)

correctly_classified_images = 0
verified_images = 0
total_images = 0

def normalize(image, means, stds):
    if(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    else:
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

        

if(dataset=='mnist'):
    csvfile = open('../data/mnist_test.csv', 'r') 
    tests = csv.reader(csvfile, delimiter=',')
else:
    csvfile = open('../data/cifar10_test.csv', 'r') 
    tests = csv.reader(csvfile, delimiter=',')

for test in tests:
    
    if(dataset=='mnist'):
        image= np.float64(test[1:len(test)])/np.float64(255)
    else:
        if(is_trained_with_pytorch):          
            image= (np.float64(test[1:len(test)])/np.float64(255))
        else:
            image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5
   
    
    specLB = np.copy(image)
    specUB = np.copy(image)
   
    if(is_trained_with_pytorch):
        normalize(specLB, means, stds)
        normalize(specUB, means, stds)
   
    label,_,_,_ = eran.analyze_box(specLB, specUB, 'deepzono', args.timeout_lp, args.timeout_milp)
    if(label == int(test[0])):
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
            normalize(specLB, means, stds)
            normalize(specUB, means, stds)
        start = time.time()
        perturbed_label, nn, nlb, nub = eran.analyze_box(specLB, specUB, domain, args.timeout_lp, args.timeout_milp)
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
                   cex_label,_,_,_ = eran.analyze_box(adv_image, adv_image, 'deepzono', args.timeout_lp, args.timeout_milp)
                   if(cex_label!=label):
                       print("adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
               
        correctly_classified_images +=1    
        end = time.time()
        print(end - start)
    else:
        print("img",total_images,"not considered, correct_label", int(test[0]), "classified label ", label)
    total_images += 1

print('analysis precision ',verified_images,'/ ', correctly_classified_images)

