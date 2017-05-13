
# coding: utf-8
import pandas as pd
import pickle
import gzip
import numpy as np
from scipy.optimize import minimize
from datetime import timedelta
import tensorflow as tf
from datetime import timedelta
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sys
import json
from dateutil.parser import parse

# def read_data_wmata(inputfile,scaletimeunit='D',precision=4,_e=1.667e-6,num_users=10):
#     with gzip.open(inputfile) as f:
#         cnt = 0
#         user_events = list()
#         user_ids= list()
#         empirical_counts=list()
#         for idx,line in enumerate(f):
#             if idx >= num_users:
#                 break

#             _user_events=list()
#             _tmp=json.loads(line.decode('utf-8'))
#             _user_events=pd.Series(sorted([parse(_dt) for _dt in _tmp['arrivalTimes']]))
#             user_ids.append(_tmp['_id'])
#             _user_events=((_user_events - _user_events[0]).apply(lambda x:
#                         round(x/np.timedelta64(1, scaletimeunit),precision))
#                           .value_counts().sort_index())
#             _user_events=_perturb(_user_events)
#             user_events.append(_user_events)
#             empirical=pd.Series(sorted([parse(_dt) for _dt in _tmp['arrivalTimes']]))
#             empirical_counts.append(empirical.value_counts().resample(rule=scaletimeunit).sum())
            
#     user_events=pd.DataFrame(user_events)
#     mask = np.nan_to_num(~(user_events.isnull()).as_matrix())

#     return np.nan_to_num(user_events.as_matrix()),empirical_counts,mask,user_ids



def read_data_wmata(inputfile,scaletimeunit='D',precision=4,_e=1.667e-6,num_users=10,test_percentage=0.2):
    with gzip.open(inputfile) as f:
        cnt = 0
        user_events_train = list()
        user_events_test=list()
        user_ids= list()
        empirical_counts=list()
        for idx,line in enumerate(f):
            _user_events=list()
            _tmp=json.loads(line.decode('utf-8'))
            _user_events=pd.Series(sorted([parse(_dt) for _dt in _tmp['arrivalTimes']]))
            user_ids.append(_tmp['_id'])
            _user_events=((_user_events - _user_events[0]).apply(lambda x:
                        round(x/np.timedelta64(1, scaletimeunit),precision))
                          .value_counts().sort_index())
            all_user_events=_perturb(_user_events)
            train_end_idx=len(all_user_events) - int(len(all_user_events)*test_percentage)
            #print(train_end_idx,len(all_user_events))
            test_user_events=all_user_events[train_end_idx+1:]
            train_user_events=all_user_events[:train_end_idx]
            user_events_train.append(train_user_events)
            user_events_test.append(test_user_events)
            empirical=pd.Series(sorted([parse(_dt) for _dt in _tmp['arrivalTimes']]))
            empirical_counts.append(empirical.value_counts().resample(rule=scaletimeunit).sum())

    user_events_train=pd.DataFrame(user_events_train)
    mask_train = ~(user_events_train.isnull()).as_matrix()
    user_events_test=pd.DataFrame(user_events_test)
    mask_test = ~(user_events_test.isnull()).as_matrix()

    return np.nan_to_num(user_events_train.as_matrix()),np.nan_to_num(user_events_test.as_matrix()),mask_train,mask_test,empirical_counts,user_ids


def _perturb(event_list,_e=1.667e-4):
    events=list()
    for t, cnt in event_list.iteritems():
        base_time = t
        for i in range(cnt):
            #here we just add i*e_i (a very small perturbation) to avoid collisions.
            events.append(base_time + (_e*i))  
    return events

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
        temporal_diff[:, :t_i, t_i] = arrivals[:, t_i][:,None] - arrivals[:, :t_i]
        
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
    regularizer=tf.reduce_sum(alpha)+tf.reduce_sum(mu)+tf.reduce_sum(beta)
    #regularizer = tf.reduce_sum(alpha)+tf.reduce_sum(mu)+tf.reduce_sum(beta) +    tf.reduce_sum(tf.square(alpha))+tf.reduce_sum(tf.square(beta))+tf.reduce_sum(tf.square(mu))#-tf.reduce_sum(tf.abs(alpha))-tf.reduce_sum(tf.abs(beta))
    return (1-_conv)*regularizer + (_conv)*nloglik

def train(events,optim,costfunc,alpha,beta,mu,_niters,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar,mask,temporal_diff,mask_tdiff):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        _cost=list()
        _alphas=list()
        _betas=list()
        _mus=list()
        #print("Initial Vals","alpha",alpha.eval(),"beta",beta.eval())
        for i in range(_niters):
            op,c,al,bt,m=(sess.run([optim,costfunc,alpha,beta,mu],feed_dict={arrivals_tvar: np.nan_to_num(events),mask_tvar:mask, temp_diff_tvar: temporal_diff,mask_tdiff_tvar:mask_tdiff}))
            if not (i%50):
                print("Iter ",i,"cost",c,"al",al[:10],"bt",bt[:10],"m",m[:10])
            _cost.append(c)
            _betas.append(bt)
            _alphas.append(al)
            _mus.append(m)

    return _cost,_alphas,_betas,_mus
            
def plot_cost(_cost,filename):
    fig,ax=plt.subplots(1,1,figsize=(15,6))
    ax.plot(_cost)
    ax.set_title("Cost")
    fig.savefig(filename,dpi=300)

def plot_params(_alphas,_betas,_mus,filename):
    fig,ax=plt.subplots(1,1,figsize=(15,6))
    ax.plot([val[0][0] for val in _alphas])
    ax.plot([val[0][0] for val in _betas])
    ax.plot([val[0][0] for val in _mus])
    ax.set_title("Alphas, Betas & Mus")
    ax.legend(["alpha","beta","mu"])
    fig.savefig(filename,dpi=300)
    
def main(inputfile,conv=0.4,learning_rate=0.000000000001,niters=2000,costfilename="cost.pdf",paramsfilename="params.pdf",modelparamsfile="modelparams.pkl",_num_users=100,test_percentage=0.2):
    events_train,events_test,mask_train,mask_test,empirical_counts,user_ids=read_data_wmata(inputfile,num_users=_num_users,test_percentage=test_percentage)

    #events,empirical_counts=read_data(inputfile)
    #mask = np.ones_like(events)
    mask_tdiff_train, temporal_diff_train = calc_temporal_differences(events_train, mask_train)
    #Init TF vars.
    num_users =mask_train.shape[0]
    alpha = tf.Variable(tf.random_uniform([num_users, 1], dtype=tf.float64), name='alpha')
    beta = tf.Variable(tf.random_uniform([num_users, 1], dtype=tf.float64), name='beta')
    mu = tf.Variable(tf.random_uniform([num_users, 1], dtype=tf.float64), name='mu')
    T_max = mask_tdiff_train.shape[1]
    arrivals_tvar = tf.placeholder(tf.float64, [None, T_max])
    mask_tvar = tf.placeholder(tf.float64, [None, T_max])
    temp_diff_tvar = tf.placeholder(tf.float64,[None, T_max, T_max])
    mask_tdiff_tvar = tf.placeholder(tf.float64, [None, T_max, T_max])

    #Evaluation.
    costfunc = cost_func(conv,alpha,beta,mu,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(costfunc)
    _c,_al,_bt,_mu=train(events_train,optimizer,costfunc,alpha,beta,mu,niters,temp_diff_tvar,mask_tdiff_tvar,arrivals_tvar,mask_tvar,mask_train,temporal_diff_train,mask_tdiff_train)

    _out={'cost':_c,'alphas':np.hstack(_al),'betas':np.hstack(_bt),'mus':np.hstack(_mu),'events_train':events_train,'events_test':events_test,'empirical_counts':empirical_counts,'user_ids':user_ids,'mask_train':mask_train,'mask_test':mask_test,'args':{'conv':conv,'test_percentage':test_percentage,'num_users':num_users}}
    #modelparamsfile+="_"+str(niters)+"_"+str(num_users)+".pkl"
    with open(modelparamsfile,"wb") as f:
        pickle.dump(_out,f)

if __name__=="__main__":
    #inputfile="/home/sathappan/workspace/time2event/hawkes/data/all_trades.csv"
    inputfile="/Users/nikhil/phd/urban_computing/datasets/wmata/wmata_2015_2016/user_timeseries_sorted_100.gz"
    #inputfile="data/all_trades.csv"
    main(inputfile,conv=float(sys.argv[1]),niters=int(sys.argv[2]),_num_users=int(sys.argv[3]),test_percentage=float(sys.argv[4]))
