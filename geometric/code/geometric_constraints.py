
from geometric_constraints_h import *
from ctypes import *


def get_transform_attack_container(config):
    get_transforms_attack_container_cpp = geometric_api.getTransformAttackContainer
    try:
        get_transforms_attack_container_cpp = geometric_api.getTransformAttackContainer
        print(2)
        get_transforms_attack_container_cpp.restype = TransformAttackContainer
        print(3)
        get_transforms_attack_container_cpp.argtypes = c_char_p
        print(4)
        transforms_attack_container = get_transforms_attack_container_cpp(config)
        print(5)
    except:
        print('get_transforms_attack_container did not work')
    return transforms_attack_container