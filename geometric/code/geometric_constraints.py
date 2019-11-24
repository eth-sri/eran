
from geometric_constraints_h import *
from ctypes import *
from numpy.ctypeslib import ndpointer


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


def get_transformations(container):
    try:
        get_transformation_dimensions_cpp = geometric_api.getTransformDimension
        get_transformation_dimensions_cpp.restype = ndpointer(c_double, flags="C_CONTIGUOUS")
        get_transformation_dimensions_cpp.argtypes = [TransformAttackContainerPtr]
        dims = get_transformation_dimensions_cpp(container)
    except:
        print('get_transformation_dimensions did not work')
    return dims


def get_transformation_dimensions(container):
    try:
        get_transformation_dimensions_cpp = geometric_api.getTransformDimension
        get_transformation_dimensions_cpp.restype = (c_int * 2)
        get_transformation_dimensions_cpp.argtypes = [TransformAttackContainerPtr]
        dims = get_transformation_dimensions_cpp(container)
    except:
        print('get_transformation_dimensions did not work')
    return dims


def get_attack_params(container):
    try:
        get_attack_params_cpp = geometric_api.get_attack_params
        get_attack_params_cpp.restype = POINTER(c_double)
        get_attack_params_cpp.argtypes = [TransformAttackContainerPtr]
        pointer = get_attack_params_cpp(container)
    except:
        print('get_attack_params did not work')
    length = get_attack_params_dim(container)
    return pointer[:length]


def get_attack_params_dim(container):
    try:
        get_attack_params_dim_cpp = geometric_api.get_attack_params_dim
        get_attack_params_dim_cpp.restype = c_int
        get_attack_params_dim_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_attack_params_dim_cpp(container)
    except:
        print('get_attack_params_dim did not work')
    return dim


def get_attack_images(container):
    try:
        get_attack_images_cpp = geometric_api.get_attack_images
        get_attack_images_cpp.restype = POINTER(c_double)
        get_attack_images_cpp.argtypes = [TransformAttackContainerPtr]
        pointer = get_attack_images_cpp(container)
    except:
        print('get_attack_images did not work')
    length = get_attack_images_dim(container)
    return pointer[:length]


def get_attack_images_dim(container):
    try:
        get_attack_images_dim_cpp = geometric_api.get_attack_images_dim
        get_attack_images_dim_cpp.restype = c_int
        get_attack_images_dim_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_attack_images_dim_cpp(container)
    except:
        print('get_attack_images_dim did not work')
    return dim


def get_attack_dimensions(container):
    try:
        get_attack_dimensions_cpp = geometric_api.getAttackDimension
        get_attack_dimensions_cpp.restype = (c_int * 2)
        get_attack_dimensions_cpp.argtypes = [TransformAttackContainerPtr]
        dims = get_attack_dimensions_cpp(container)
    except:
        print('get_attack_dimensions did not work')
    return dims