#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from util import createNearestNeighbor, createAgeGraph,normalize
from cost_functions import *
import cost_functions
import time
import client_parser
import time
from libraries.mutation_signatures_visualization import sbs_signature_plot
import matplotlib.pyplot as plt 

np.random.seed(0)
HIDDEN_PERCENT = 0.3

data,patients,categories = client_parser.parse_counts("data/brca_data.tsv")
weight_matrix = client_parser.matrix("data/brca_graph.tsv",patients)


shape = data.shape
all_pairs = [(i, j) for i in range(shape[0]) for j in
             range(shape[1])]
np.random.shuffle(all_pairs)
hidden_pairs = all_pairs[:int(round(HIDDEN_PERCENT
                              * len(all_pairs)))]

np.set_printoptions(suppress=True)

print("Created graph")

def create_mask(A_orig):
    A_orig_df = pd.DataFrame(A_orig)

    A_df_masked = A_orig_df.copy()

    for (i, j) in hidden_pairs:
        A_df_masked.iloc[i, j] = np.NAN

    return A_df_masked
V_masked = create_mask(data)

def init_W_H(shape,rank=5):
    temp_H = np.abs(np.random.randn(rank, shape[1])).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.abs(np.random.randn(shape[0], rank)).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H = tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    
    return W,H

def get_clip(W,H):
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)
    return clip

def find_results(rank,f,use_random=False,lr = 0.0001):
    global weight_matrix
    
    bool_mask = V_masked.notnull()
    
    tf_mask = tf.Variable(bool_mask.values)
    V = tf.constant(V_masked.values)
        
    laplacian_matrix = laplacian(weight_matrix).astype(np.float32)
    W,H = init_W_H(shape,rank=rank)
    WH = tf.matmul(W, H)
    L = tf.constant(laplacian_matrix)
    WTLW = tf.matmul(tf.matmul(tf.transpose(W),L),W)
    
    cost = f(V,tf_mask,WH,WTLW)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()
    clip = get_clip(W,H)

    sess = tf.Session()
    sess.run(init)

    """
    graph_cost = sess.run(tf.linalg.trace(WTLW))
    target_percent = 0.1
    actual_percent = graph_cost/sess.run(cost)
    lamb = target_percent/actual_percent
    print(lamb)
    """
    
    i = 0
    previous_cost = sess.run(cost)
    
    while np.isfinite(sess.run(cost)) and (i<=2000): #or previous_cost-sess.run(cost)>10):
        previous_cost = sess.run(cost)
        sess.run(train_step)
        sess.run(clip)
        i+=1

    learnt_W = sess.run(W)
    learnt_H = sess.run(H)

        
    difference = np.linalg.norm((data- learnt_W.dot(learnt_H))[~bool_mask])
    sbs_signature_plot(pd.DataFrame(learnt_H,columns=categories))
    plt.savefig(str(difference)+".jpg")

    return difference

def run_trials(rank,f,use_random=False,lr=.0001):
    return np.min([find_results(rank,f,use_random=use_random,lr=lr) for i in range(5)])


def find_lr(lamb):
    possible = [-1,-5,-9]
    distance = 4
    cost_functions.lamb = lamb
    for i in range(3):
        results = [run_trials(5,frobenius,lr=10**j) for j in possible]
        best_score = possible[results.index(min(results))]
        distance/=2
        possible = [best_score-distance,best_score,best_score+distance]
        print(possible)

    return 10**possible[1]

#finding learning rate
t = time.time()
lr = 10**-5 #find_lr(0)
cost_functions.lamb = 0   
baseline = run_trials(5,frobenius,lr=lr)
scores = []
print(baseline)

"""
for i in [0,1,2,3,4,5,6,7,8,9,10]:
    cost_functions.lamb = 10**i
    lr = find_lr(10**i)
    scores.append((run_trials(5,frobenius,lr=lr),i))
    print(scores)

for i in range(len(scores)):
    scores[i] = (scores[i][0]/baseline,scores[i][1])

print(min(scores,key=lambda x: x[0]))
print(time.time()-t)"""
