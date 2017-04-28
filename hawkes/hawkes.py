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

import numpy as np
from numpy.random import exponential as exponential_rng
from commons.commons import exponential_kernel, convert_to_numpy
from scipy.optimize import minimize


class UniVariateHawkes(object):
    def __init__(self, lambda_0=None, alpha_0=None, beta=None):
        """
        """
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta = beta

    def loglikelihood(self, eventHistory, lambda_0,
                      alpha_0, beta):
        """
        calc negative log likelihood of hawkes process
        Params:
            eventHistory - numpy array of inter-arrival times per user
            lambda_0 - numpy array of base intensity per user
            alpha_0 - numpy array influence weight of history per user
            beta - time_scale parameter for exponential kernel per user
        """

        eventHistory, lambda_0, alpha_0, beta = convert_to_numpy(eventHistory, lambda_0,
                                                                 alpha_0, beta)

        t_deltas = np.c_[np.zeros(eventHistory.shape[0]),
                         exponential_kernel(np.diff(eventHistory), scale=-1 * beta.T)]

        A_i = np.zeros_like(eventHistory)
        for i in range(1, eventHistory.shape[1]):
            A_i[:, i] = t_deltas[:, i] * (1 + A_i[:, i - 1])

        part1 = np.sum(np.log(lambda_0.T + alpha_0.T * A_i), axis=1)[:, np.newaxis]
        part2 = lambda_0.T * eventHistory[:, -1, None]
        part3 = np.sum(exponential_kernel(eventHistory[:, -1][:, np.newaxis] - eventHistory,
                                          scale=-1 * beta.T) - 1, axis=1)

        part3 = (alpha_0 / beta).T * part3[:, np.newaxis]
        return np.sum(part1 - part2 + part3)

    def simulate_ogataThinning(self, lambda_0=None, alpha_0=None,
                               beta=None, maxPoints=None):
        """
        Simulate hawkes process using the Modified Ogatta thinning process
        Reference:  Hawkes Process, P.J Laub et.al., 2015
        Params:
            lambda_0 - base intensity to use
            alpha_0 - influence weight for history to use
            beta - exponential kernel parameter to use
        """

        assert all([lambda_0, alpha_0, beta]), """lambda_0 alpha_0 and scale
        parameters are necessary to perform simulation"""

        lambda_0, alpha_0, beta = convert_to_numpy(lambda_0, alpha_0, beta)

        T = 500.0
        t = 0
        events = np.zeros((lambda_0.shape[1], maxPoints))
        epsilon = 1e-6
        enum = 0
        while t < T and enum < maxPoints:
            m = self.get_intensity_upperBound(lambda_0, alpha_0, beta,
                                              t + epsilon, events[:, :enum])
            t += -np.log(np.random.random_sample()) / m
            #u = np.random.random_sample() * m
            u = np.random.uniform(0,m) 
            if t < T and u <= self.get_intensity_upperBound(lambda_0, alpha_0,
                                                            beta, t, events[:, :enum]):
                events[:, enum] = t
                enum += 1

        return events

    def get_intensity_upperBound(self, lambda_0, alpha_0, beta,
                                 current_time, eventHistory):
        """
        get intensity upperbound
        """
        t = current_time
        A_i = np.sum(exponential_kernel(delta=(t - eventHistory),
                                        scale=(-1 * beta)))

        return lambda_0 + np.dot(alpha_0.T, A_i)

    def calc_conditionIntensity(self, lambda_0, alpha_0,
                                beta, timePoints, eventHistory):
        """
        calculate the  Conditional Intensity at each time interval
        \lambda (t) = \lambda_0 + \alpha * \sum_{i=0}^{t} (\exp{-\beta (t-t_i)})
        Params:
            lambda_0 - base intensity per user
            alpha_0, beta - exponential kernel parameter per user
        """
        def hawkes_cif(t):
            history = eventHistory[eventHistory < t]
            return (lambda_0 +
                    alpha_0 * np.sum(exponential_kernel(t - history,
                                                        scale=-beta))
                    )

        return np.array([hawkes_cif(t) for t in timePoints])

    def calc_compensator(self, lambda_0, alpha_0, beta, timePoints,
                         eventHistory):
        """
        calculate the value of the compensator at the given time points
        compensator is defined as  \Lambda (t) = \int_0^t \lambda (t) dt
        """
        if not isinstance(eventHistory, np.ndarray):
            raise Exception("Event History should be an numpy array")

        def compensator(t):
            history = eventHistory[eventHistory < t]
            part = (np.sum(exponential_kernel(t - history, scale=-beta) - 1))
            return lambda_0 * t - (alpha_0 / beta) * part

        return [compensator(t) for t in timePoints]

    def fit(self, init_params, data, method='Nelder-Mead'):
        bounds = ((0, None),)*(data.shape[0]*len(init_params))
        mu, alpha, beta = init_params

        constraint1 = [{"type": "ineq", "fun": lambda x: x[2] - x[1]}]

        def cost_func(params, data):
            try:
                mu, alpha, beta = params
            except:
                mu, alpha, beta = np.hsplit(params.reshape((data.shape[0],3)), 3)
                mu = mu.T
                alpha = alpha.T
                beta = beta.T
            return (-self.loglikelihood(eventHistory=data,
                                        lambda_0=mu, alpha_0=alpha,
                                        beta=beta)
                    + (mu ** 2 + alpha ** 2 + beta ** 2).sum())

        return minimize(cost_func, (mu, alpha, beta), args=(data,), bounds=bounds,
                        constraints=constraint1, method=method)


class MultiVariateHawkes(object):
    def __init__(self):
        """
        """
        pass

    def neg_loglikelihood(self, eventHistory, lambda_0, alpha_0, beta):
        """
        """
        pass

    def simulate(self):
        pass
