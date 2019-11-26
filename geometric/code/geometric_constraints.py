
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


def get_transformation_dim_0(container):
    try:
        get_transformations_dim_0_cpp = geometric_api.get_transformations_dim_0
        get_transformations_dim_0_cpp.restype = c_int
        get_transformations_dim_0_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_transformations_dim_0_cpp(container)
    except:
        print('get_transformation_dim_0 did not work')
    return dim


def get_transformation_dim_1(container, dim_0):
    try:
        get_transformations_dim_1_cpp = geometric_api.get_transformations_dim_1
        get_transformations_dim_1_cpp.restype = POINTER(c_int)
        get_transformations_dim_1_cpp.argtypes = [TransformAttackContainerPtr]
        pointer = get_transformations_dim_1_cpp(container)
    except:
        print('get_transformation_dim_1 did not work')
    return pointer[:dim_0]


def get_transformations(container):
    try:
        get_transformation_dimensions_cpp = geometric_api.get_transformations
        get_transformation_dimensions_cpp.restype = POINTER(POINTER(c_double))
        get_transformation_dimensions_cpp.argtypes = [TransformAttackContainerPtr]
        pointer = get_transformation_dimensions_cpp(container)
    except:
        print('get_transformations did not work')
    dim_0 = get_transformation_dim_0(container)
    dim_1 = get_transformation_dim_1(container, dim_0)
    return [p[:dim_1[i]] for i, p in enumerate(pointer[:dim_0])]


def get_attack_params_dim_0(container):
    try:
        get_attack_params_dim_0_cpp = geometric_api.get_attack_params_dim_0
        get_attack_params_dim_0_cpp.restype = c_int
        get_attack_params_dim_0_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_attack_params_dim_0_cpp(container)
    except:
        print('get_attack_params_dim did not work')
    return dim


def get_attack_params_dim_1(container):
    try:
        get_attack_params_dim_1_cpp = geometric_api.get_attack_params_dim_1
        get_attack_params_dim_1_cpp.restype = c_int
        get_attack_params_dim_1_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_attack_params_dim_1_cpp(container)
    except:
        print('get_attack_params_dim did not work')
    return dim


def get_attack_params(container):
    try:
        get_attack_params_cpp = geometric_api.get_attack_params
        get_attack_params_cpp.restype = POINTER(POINTER(c_double))
        get_attack_params_cpp.argtypes = [TransformAttackContainerPtr]
        pointer = get_attack_params_cpp(container)
    except:
        print('get_attack_params did not work')
    dim_0 = get_attack_params_dim_0(container)
    dim_1 = get_attack_params_dim_1(container)
    return [p[:dim_1] for p in pointer[:dim_0]]


def get_attack_images_dim_0(container):
    try:
        get_attack_images_dim_0_cpp = geometric_api.get_attack_images_dim_0
        get_attack_images_dim_0_cpp.restype = c_int
        get_attack_images_dim_0_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_attack_images_dim_0_cpp(container)
    except:
        print('get_attack_images_dim did not work')
    return dim


def get_attack_images_dim_1(container):
    try:
        get_attack_images_dim_1_cpp = geometric_api.get_attack_images_dim_1
        get_attack_images_dim_1_cpp.restype = c_int
        get_attack_images_dim_1_cpp.argtypes = [TransformAttackContainerPtr]
        dim = get_attack_images_dim_1_cpp(container)
    except:
        print('get_attack_images_dim did not work')
    return dim


def get_attack_images(container):
    try:
        get_attack_images_cpp = geometric_api.get_attack_images
        get_attack_images_cpp.restype = POINTER(POINTER(c_double))
        get_attack_images_cpp.argtypes = [TransformAttackContainerPtr]
        pointer = get_attack_images_cpp(container)
    except:
        print('get_attack_images did not work')
    dim_0 = get_attack_images_dim_0(container)
    dim_1 = get_attack_images_dim_1(container)
    return [p[:dim_1] for p in pointer[:dim_0]]