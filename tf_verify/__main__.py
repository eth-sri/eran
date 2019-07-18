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

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .pyt, .tf and .meta')
parser.add_argument('--epsilon', type=float, default=0, help='the epsilon for L_infinity perturbation')
#parser.add_argument('--specnumber', type=int, default=9, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=None, help='the domain name can be either deepzono, refinezono or deeppoly')
parser.add_argument('--dataset', type=str, default=None, help='the dataset, can be either mnist, cifar10, or acasxu')
parser.add_argument('--complete', type=str2bool, default=False,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=1,  help='timeout for the LP solver')
parser.add_argument('--timeout_milp', type=float, default=1,  help='timeout for the MILP solver')
parser.add_argument('--use_area_heuristic', type=str2bool, default=True,  help='whether to use area heuristic for the DeepPoly ReLU approximation')

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
if((dataset!='mnist') and (dataset!='cifar10') and (dataset!='acasxu')):
    print("only mnist, cifar10, and acasxu datasets are supported")
    exit(1)


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
    elif (dataset=='cifar10'):
        num_pixels = 3072
    else:
        num_pixels = 5
    model, is_conv, means, stds = read_net(netname, num_pixels, is_trained_with_pytorch)
    eran = ERAN(model)

correctly_classified_images = 0
verified_images = 0
total_images = 0

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
        

if(dataset=='mnist'):
    csvfile = open('../data/mnist_test.csv', 'r') 
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='cifar10'):
    csvfile = open('../data/cifar10_test.csv', 'r') 
    tests = csv.reader(csvfile, delimiter=',')
else:
    specfile = '../data/acasxu/specs/acasxu_prop' + str(specnumber) +'_spec.txt'
    tests = open(specfile, 'r').read()
    

if dataset=='acasxu':
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
    _,nn,_,_ = eran.analyze_box(specLB, specUB, 'deepzono', args.timeout_lp, args.timeout_milp, args.use_area_heuristic, specnumber)
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

else:
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
        
        label,nn,nlb,nub = eran.analyze_box(specLB, specUB, domain, args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
        #print("nlb ", nlb[len(nlb)-1])
        #print("nub ",nub[len(nub)-1])
            #print("nlb ", nlb[3])
            #print("nub ",nub[3])
        #for number in range(len(nub)):
        #    for element in range(len(nub[number])):
        #        if(nub[number][element]<=0):
        #            print('False')
        #        else:
        #            print('True')
                    
        #print("concrete ", nlb[len(nlb)-1])
        #if(label == int(test[0])):
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
            perturbed_label, _, nlb, nub = eran.analyze_box(specLB, specUB, domain, args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
            #print( "nlb ", nlb[len(nlb)-1], " nub ", nub[len(nub)-1])
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
                               denormalize(adv_image, means, stds)
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

