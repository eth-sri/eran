import csv

def get_tests(dataset):
    if (dataset == 'acasxu'):
        specfile = './acasxu/specs/acasxu_prop' + str(specnumber) + '_spec.txt'
        tests = open(specfile, 'r').read()
    else:
        csvfile = open('./{}_test.csv'.format(dataset), 'r')
        tests = csv.reader(csvfile, delimiter=',')
    return tests

zonotope_file = open('./zonotope_file', 'w+')

epsilon = 0.026
tests = get_tests('mnist')
image_number = 0

for index, test in enumerate(tests):
    if index < image_number:
        continue
    test = test[1:]
    zonotope_file.write(str(len(test)) + ' ' + str(1+len(test))+'\n')
    for dim, a_0 in enumerate(test):
        current_epsilon = epsilon
        #normalize to [0, 1]
        normalized = int(a_0)/255.0
        if normalized < epsilon:
            current_epsilon = (epsilon + normalized)/2
            normalized = current_epsilon
        elif normalized > (1 - epsilon):
            current_epsilon = (epsilon + (1-normalized))/2
            normalized = 1 - current_epsilon
        zonotope_file.write(str(normalized) + ' ')
        for i in range(len(test)):
            if i == dim:
                zonotope_file.write(str(current_epsilon) + ' ')
            else:
                zonotope_file.write('0 ')
        zonotope_file.write('\n')
    break

zonotope_file.close()