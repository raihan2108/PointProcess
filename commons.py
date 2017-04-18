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
