from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.stats import normaltest, zscore
import tensorflow as tf
from tensorflow.contrib.util import constant_value
import numpy as np
from math import sqrt
import sys
import re


# There exists an option to initialize matrix as Identity
# But trying to initialize with it result in error
# Uncomment when this moment is clarified
# def test_identity(arr):
#     s = arr.shape
#     if (s[0] >= 2 and s[1] >= 2 and
#         (s[0] > s[1] and np.allclose(arr[:s[1]], np.eye(s[1]))
#          or np.allclose(arr[:,:s[0]], np.eye(s[0])))):
#         return True
#     return False


def test_orthogonal(arr):
    s = arr.shape
    if (s[0] >= 2 and s[1] >= 2 and
        (s[0] < s[1] and np.allclose(arr[0].dot(arr[1]), [0]) or
         np.allclose(arr[:,0].dot(arr[:,1]), [0]))):
        return True
    return False


def ecmp(epsilon):
    def cmp(a, b):
        return abs(a - b) < epsilon
    return cmp


def detect_init(_arr, alpha, epsilon):
    '''
    This function uses indicative properties of each available initializer
    It assumes that initializers created with default values
    See https://www.tensorflow.org/api_docs/python/tf/keras/initializers
    Parameters:
    _arr: numpy.ndarray
    alpha: rate of normal testing
    epsilon: rate of equality
    '''
    if len(_arr.shape) == 1:
        arr = _arr.reshape((_arr.shape[0], 1))
    elif len(_arr.shape) == 2:
        # if test_identity(_arr):
        #     return "Identity"
        if test_orthogonal(_arr):
            return "Orthogonal"

        arr = _arr[:]
    else:
        raise TypeError('Expected numpy array with 1 or 2 dimensions, got ' + len(_arr.shape))

    flat_arr = arr.flatten()
    is_constant = np.unique(flat_arr).shape[0] <= 2
    is_normal = normaltest(flat_arr)[1] > alpha
    is_uniform = not (is_constant or is_normal)
    fan_in, fan_out = arr.shape
    equal = ecmp(epsilon)
    
    if is_constant:
        probe_min = float(flat_arr.min())
        probe_max = float(flat_arr.max())
        if probe_max == 0.0:
            return "Zeros"
        elif probe_min == 1.0:
            return "Ones"

        # Fabrik initializes constant matrices with 0 and 1
        return "constant"

    if is_normal:
        probe_edge_value = flat_arr.min()
        std = flat_arr.std()
        z = zscore(flat_arr)
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        truncnorm = .87962566103423978

        # @TODO: find a proper way to detect truncated normal distributions
        if equal(std, sqrt(2 / (fan_in + fan_out)) / truncnorm):
            return "glorot_normal"
        if equal(std, sqrt(2 / fan_in) / truncnorm):
            return "he_normal"
        if equal(std, sqrt(1 / fan_in) / truncnorm):
            return "VarianceScaling"
        elif equal(std, truncnorm):
            return "TruncatedNormal"

        return "RandomNormal"

    if is_uniform:
        m_limit = flat_arr.min()
        limit = flat_arr.max()
        
        glorot = sqrt(6 / (fan_in + fan_out))
        he = sqrt(6 / fan_in)
        lecun = sqrt(3 / fan_in)

        if equal(limit, glorot) and equal(m_limit, -glorot):
            return "glorot_uniform"
        if equal(limit, he) and equal(m_limit, -he):
            return "he_uniform"
        if equal(limit, sqrt(3 / lecun)) and equal(m_limit, -lecun):
            return "lecun_uniform"
        
    return "RandomUniform"

