
from ctypes import *
geometric_api = CDLL("libgeometric.so")


class TransformAttackContainer(Structure):

    _fields_ = [('transforms', c_double * 10), ('attacks', c_double * 10)]