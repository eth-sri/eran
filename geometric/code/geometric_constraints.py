
from geometric_constraints_h import *
from ctypes import *


def get_transform_attack_container(config):
    try:
        c_config = c_char_p(bytes(config, 'utf-8'))
        get_transforms_attack_container_cpp = geometric_api.getTransformAttackContainer
        get_transforms_attack_container_cpp.restype = TransformAttackContainerPtr
        get_transforms_attack_container_cpp.argtypes = [c_char_p]
        transforms_attack_container = get_transforms_attack_container_cpp(c_config)
    except:
        print('get_transforms_attack_container did not work')
    return transforms_attack_container


def set_transform_attack_for(container, i):
    try:
        get_transforms_attack_container_cpp = geometric_api.setTransformationsAndAttacksFor
        get_transforms_attack_container_cpp.restype = None
        get_transforms_attack_container_cpp.argtypes = [TransformAttackContainerPtr, c_int]
        transforms_attack_container = get_transforms_attack_container_cpp(container, i)
    except:
        print('set_transform_attack_for did not work')
    return transforms_attack_container