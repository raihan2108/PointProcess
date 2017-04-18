#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

from __future__ import division

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import tensorflow as tf
import numpy as np
from .commons import tflow_diff as diff


## Assuming number of users is 32
num_users = 1

## All parameters are column vectors with size equal to number of users
def init_variables(num_users):
    alpha = tf.Variable(tf.zeros([num_users, 1]))
    beta = tf.Variable(tf.ones([num_users, 1]))
    mu = tf.Variable(tf.zeros([num_users, 1]))
    test = tf.zeros_like(arrivals)
    return alpha, beta, mu, test

def loglikelihood():
    tdiff = diff(arrivals) * mask[:, 1:]
    A_i = tf.zeros_like(arrivals)
    for i in range(tdiff.shape[0]):
        A_i[:, i + 1] = tdiff[:, i] * (1 + A[:, i ])
    
    part1 = tf.reduce_sum(tf.log(mu + alpha * A_i), axis=1)
        
    part2 = mu * T_max
    part3 = (alpha / beta)* tf.reduce_sum(tf.exp(-beta * ((T_max - arrivals)*mask)) - tf.constant(1.0, dtype=tf.float32))
    
    return part1 - part2 + part3

def cost_func():
    return -loglikelihood()

def run(events):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
    sess.run(init)
    negloglik = cost_func()
    sess.run(negloglik, feed_dict={arrivals: events, mask: np.ones_like(events)})
