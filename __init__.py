#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    Basic Hawkes Process package
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"


from abc import ABCMeta

class PointProcess:
    __metaclass__ = ABCMeta

    def neg_loglikelihood(self):
        """
        calculate negative log likelihood
        """
        pass

    def calc_gradient(self):
        """
        calc gradient
        """

    def simulate(self):
        """
        simulate point process
        """

