
# coding: utf-8
import pandas as pd
import gzip
import numpy as np
from scipy.optimize import minimize
from datetime import timedelta
import tensorflow as tf
from datetime import timedelta
from scipy.sparse import csr_matrix
#import matplotlib.pyplot as plt
import sys

def read_data(inputfile,_e=1.667e-6):
    """
        Read inputfile , process difference between first event and each subsequent event.
        Assumption is that events are in sorted order by datetime.
    """
    
    df = pd.read_csv(inputfile)
    df.t = pd.to_datetime(df.t)
    pp = (df.t - df.t[0]).apply(lambda x: x/np.timedelta64(1, 'm')).value_counts()
    events = []
    e = _e
    for t, cnt in pp.iteritems():
        base_time = t
        for i in range(cnt):
            #here we just add i*e_i (a very small perturbation) to avoid collisions.
            events.append(base_time + e*i)  

    events = np.array(sorted(events))
    empirical = df.t
    empirical_counts = empirical.value_counts().resample(rule='1min').sum()
    events = events[None, :]   #None operation adds an extra dimention of size 1 to events.
    return events,empirical_counts

def calc_temporal_differences(arrivals, mask):
    #U X T X T tensor.
    temporal_diff = np.zeros((arrivals.shape[0], arrivals.shape[1], arrivals.shape[1]))
    
    for t_i in range(arrivals.shape[1]):
        temporal_diff[:, :t_i, t_i] = arrivals[:, t_i] - arrivals[:, :t_i]
        
    mask_tempdiff = (temporal_diff != 0) 
    
    return mask_tempdiff, temporal_diff

def loglikelihood(alpha,beta,mu,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar):
    
    hist_effects = tf.reduce_sum((tf.exp(-beta[:, :, None] * temp_diff_tvar))                                 *mask_tdiff_tvar, axis=2)
    part1 = tf.reduce_sum(tf.log(mu + alpha * hist_effects), axis=1)
    T_max = tf.reduce_max(arrivals_tvar*mask_tvar, axis=1)[:, None]
    part2 = mu * T_max
    tmp = tf.exp(-beta * ((T_max - arrivals_tvar)*mask_tvar))
    p3_tmp = tf.reduce_sum(tmp - tf.constant(1.0, dtype=tf.float64), axis=1)
    part3 = (alpha / beta)* p3_tmp[:, None]    
    return tf.reduce_sum(part1 - part2 + part3)

def cost_func(_conv,alpha,beta,mu,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar):
    nloglik = -loglikelihood(alpha,beta,mu,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar)
    regularizer = tf.reduce_sum(alpha)+tf.reduce_sum(mu)+tf.reduce_sum(beta) +    tf.reduce_sum(tf.square(alpha))+tf.reduce_sum(tf.square(beta))+tf.reduce_sum(tf.square(mu))#-tf.reduce_sum(tf.abs(alpha))-tf.reduce_sum(tf.abs(beta))
    return (1-_conv)*regularizer + (_conv)*nloglik

def train(events,optim,costfunc,alpha,beta,mu,_niters,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar,mask,temporal_diff,mask_tdiff):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        _cost=list()
        _alphas=list()
        _betas=list()
        _mus=list()
        print("Initial Vals","alpha",alpha.eval(),"beta",beta.eval())
        for i in range(_niters):
            op,c,al,bt,m=(sess.run([optim,costfunc,alpha,beta,mu],feed_dict={arrivals_tvar: np.nan_to_num(events),                                   mask_tvar:mask, temp_diff_tvar: temporal_diff,                                   mask_tdiff_tvar:mask_tdiff}))
            if not (i%50):
                print("Iter ",i,"cost",c,"al",al,"bt",bt,"m",m)
            _cost.append(c)
            _betas.append(bt)
            _alphas.append(al)
            _mus.append(m)

    return _cost,_alphas,_betas,_mus
            
def plot_cost(filename):
    plt.plot(_cost)
    plt.title("Cost")
    plt.savefig(filename,dpi=300)

def plot_params(filename):
    plt.plot([val[0][0] for val in _alphas])
    plt.plot([val[0][0] for val in _betas])
    plt.plot([val[0][0] for val in _mus])
    plt.title("Alphas, Betas & Mus")
    plt.legend(["alpha","beta","mu"])
    plt.savefig(filename,dpi=300)
    
def main(inputfile,conv=0.4,learning_rate=0.000001,niters=2000,costfilename="cost.txt",paramsfilename="params.txt"):
    events,empirical_counts=read_data(inputfile)
    mask = np.ones_like(events)
    mask_tdiff, temporal_diff = calc_temporal_differences(events, mask)
    
    #Init TF vars.
    num_users = mask.shape[0]
    alpha = tf.Variable(tf.random_uniform([num_users, 1], dtype=tf.float64), name='alpha')
    beta = tf.Variable(tf.random_uniform([num_users, 1], dtype=tf.float64), name='beta')
    mu = tf.Variable(tf.random_uniform([num_users, 1], dtype=tf.float64), name='mu')
    T_max = mask_tdiff.shape[1] 
    arrivals_tvar = tf.placeholder(tf.float64, [None, T_max])
    mask_tvar = tf.placeholder(tf.float64, [None, T_max])
    temp_diff_tvar = tf.placeholder(tf.float64,[None, T_max, T_max])
    mask_tdiff_tvar = tf.placeholder(tf.float64, [None, T_max, T_max])
    
    #Evaluation.
    costfunc = cost_func(conv,alpha,beta,mu,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(costfunc)
    _c,_al,_bt,_mu=train(events,optimizer,costfunc,alpha,beta,mu,niters,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar,mask,temporal_diff,mask_tdiff)

    with open(costfilename) as f:
	json.dump(f,_c)

    #Plot
    #plot_cost(costfilename)
    #plot_params(paramsfilename)

if __name__=="__main__":
    #inputfile="/home/sathappan/workspace/time2event/hawkes/data/all_trades.csv"
    #inputfile="/Users/nikhil/phd/urban_computing/wmata/repos/PointProcess/data/all_trades.csv"
    inputfile="data/all_trades.csv"
    main(inputfile,niters=1000)
