import re
import os.path
import numpy as np

def read_zonotope(zonotope_file):
    assert os.path.isfile(zonotope_file), 'There exists no zonotope file.'

    zonotope_read = open(zonotope_file,'r').read()
    zonotope = re.split('[, \n]+', zonotope_read)
    zonotope_height = int(zonotope[0])
    zonotope_width = int(zonotope[1])

    zonotope = [np.float64(x) for x in zonotope[2:zonotope_width*zonotope_height+2]]
    zonotope = np.array(zonotope)
    return np.reshape(zonotope, (zonotope_height, zonotope_width))