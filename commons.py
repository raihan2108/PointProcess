#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import numpy as np


def convert_to_numpy(*args):
    """
    convert all args to numpy objects
    """
    return [np.array(l, ndmin=2) for l in args]


def exponential_kernel(delta=None, scale=1):
    """
    return the exponential kernel value.
    exp(scale * (a - b))
    """
    return np.exp(delta * scale)


def tflow_diff(a, n=1, axis=-1):
    """copied shamelessly from `http://stackoverflow.com/questions/42609618/tensorflow-equivalent-to-numpy-diff anwer by Frank Dernoncourt`"""
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return diff(a[slice1]-a[slice2], n-1, axis=axis)
    else:
        return a[slice1]-a[slice2]