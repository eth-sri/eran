import multiprocessing

from enum import Enum

class Device(Enum):
    CPU = 0
    CUDA = 1


class config:
    use_milp = True # Whether to use MILP
    dyn_krelu = False # dynamically select parameter k
    use_2relu = False # use 2-relu
    use_3relu = False # use 3-relu
    numproc = multiprocessing.cpu_count() # number of processes for milp/lp/krelu
    netname = None # the network name, the extension can be only .pyt, .tf and .meta
    epsilon = 0 # the epsilon for L_infinity perturbation
    zonotope = None # file to specify the zonotope matrix
    domain = None # the domain name can be either deepzono, refinezono, deeppoly or refinepoly
    dataset = None # the dataset, can be either mnist, cifar10, or acasxu
    complete = False # flag specifying where to use complete verification or not
    timeout_lp = 1 # timeout for the LP solver
    timeout_milp = 1 # timeout for the MILP solver
    use_default_heuristic = True # whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation
    mean = None # the mean used to normalize the data with
    std = None # the standard deviation used to normalize the data with
    data_dir = None # data location for geometric analysis
    geometric_config = None # geometric config location
    num_params = 0 # Number of transformation parameters for geometric analysis
    num_tests = None # Number of images to test
    from_test = 0 # From which number to start testing
    debug = False # Whether to display debug info
    attack = False # Whether to attack in geometric analysis
    geometric = False # Whether to do geometric analysis
    specnumber = None # Acas Xu spec number
    input_box = None # input box file to use
    output_constraints = None # output constraints file to check
    device = Device.CPU # Which device Deeppoly should run on
