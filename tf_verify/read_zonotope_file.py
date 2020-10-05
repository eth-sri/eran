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
