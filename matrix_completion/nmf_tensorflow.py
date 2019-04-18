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

HIDDEN_PERCENT = 0.3

data,patients = client_parser.parse_counts("data/counts.tsv")
weight_matrix = client_parser.parse_cna("data/cna.tsv")
weight_patients = client_parser.parse_cna_patients("data/cna_patients.tsv")

used_weight_patients = []
used_patients = []
for i in range(len(patients)):
    if patients[i] in weight_patients:
        used_patients.append(i)

for i in range(len(weight_patients)):
    if weight_patients[i] in patients:
        used_weight_patients.append(i)

data = data[used_patients]
used_weight_patients = np.array(used_weight_patients)
weight_matrix = weight_matrix[used_weight_patients[:,None],
                              used_weight_patients]

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

def find_results(rank,f,k=5,use_random=False,lr = 0.0001):
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
    
    while np.isfinite(sess.run(cost)) and (i<=1000): #or previous_cost-sess.run(cost)>10):
        previous_cost = sess.run(cost)
        sess.run(train_step)
        sess.run(clip)
        i+=1

    learnt_W = sess.run(W)
    learnt_H = sess.run(H)
        
    difference = np.linalg.norm((data- learnt_W.dot(learnt_H))[~bool_mask])

    return difference

def run_trials(rank,f,k=5,use_random=False,lr=.0001):
    return np.min([find_results(rank,f,k=k,use_random=use_random,lr=lr) for i in range(5)])

lr = 10**(-5.25)
cost_functions.lamb = 0
baseline = run_trials(5,frobenius,k=5,lr=lr)
scores = []
for i in [0,0.5,1,1.5,2]:
    cost_functions.lamb = 10**i
    scores.append((run_trials(5,frobenius,k=5,lr=lr),i))

for i in range(len(scores)):
    scores[i] = (scores[i][0]/baseline,scores[i][1])
print(min(scores,key=lambda x: x[0]))

