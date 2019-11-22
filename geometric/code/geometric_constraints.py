
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
        set_transform_attack_for_cpp = geometric_api.setTransformationsAndAttacksFor
        set_transform_attack_for_cpp.restype = None
        set_transform_attack_for_cpp.argtypes = [TransformAttackContainerPtr, c_int]
        set_transform_attack_for_cpp(container, i)
    except:
        print('set_transform_attack_for did not work')


def get_transformation_dimensions(container):
    try:
        get_transformation_dimensions_cpp = geometric_api.getTransformDimension
        get_transformation_dimensions_cpp.restype = (c_int * 2)
        get_transformation_dimensions_cpp.argtypes = [TransformAttackContainerPtr]
        dims = get_transformation_dimensions_cpp(container)
    except:
        print('get_transformation_dimensions did not work')
    return dims


def get_attack_dimensions(container):
    try:
        get_attack_dimensions_cpp = geometric_api.getAttackDimension
        get_attack_dimensions_cpp.restype = (c_int * 2)
        get_attack_dimensions_cpp.argtypes = [TransformAttackContainerPtr]
        dims = get_attack_dimensions_cpp(container)
    except:
        print('get_attack_dimensions did not work')
    return dims