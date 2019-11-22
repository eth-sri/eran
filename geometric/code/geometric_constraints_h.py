
from ctypes import *
geometric_api = CDLL("libgeometric.so")


class TransformAttackContainer(Structure):

    _fields_ = []

TransformAttackContainerPtr = POINTER(TransformAttackContainer)