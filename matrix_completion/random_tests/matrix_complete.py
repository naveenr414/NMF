#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..") # Adds higher directory to python modules path.

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

np.random.seed(0)

data2,patients,categories = client_parser.parse_counts("../data/brca_data.tsv")
weight_matrix = client_parser.matrix("../data/brca_graph.tsv",patients).astype(np.float32)

data = weight_matrix
shape = data.shape

np.set_printoptions(suppress=True)

print("Created graph")

def create_mask(A_orig):
    A_orig_df = pd.DataFrame(A_orig)

    A_df_masked = A_orig_df.copy()
    A_df_masked[data == 0] = np.NAN

    return A_df_masked

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

def matrix_complete(V,rank,lr = 0.0001):
    V_masked = create_mask(V)

    bool_mask = V_masked.notnull()

    tf_mask = tf.Variable(bool_mask.values)
    V = tf.constant(V_masked.values)

    W,H = init_W_H(shape,rank=rank)
    WH = tf.matmul(W, H)

    cost = tf.reduce_sum(tf.pow(tf.boolean_mask(V, tf_mask)
                         - tf.boolean_mask(WH, tf_mask), 2))
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

    return learnt_W.dot(learnt_H)

V_masked = create_mask(data)

#finding learning rate
t = time.time()
lr = 10**-5 #find_lr(0)
V = matrix_complete(data,5)

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
