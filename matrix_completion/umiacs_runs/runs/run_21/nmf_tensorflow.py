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
from copy import deepcopy
import client_parser
import time
import sys
import networkx as nx  
from sklearn.preprocessing import Imputer

HIDDEN_PERCENT = 0.2
permute_randomly = True
k_nearest = False
shuffle_k = False

data,patients = client_parser.parse_counts("data/brca_data.tsv")
#Fix weight_matrix
weight_matrix = client_parser.matrix("data/brca_graph.tsv",patients)
weight_matrix[weight_matrix == 0] = np.NAN
imp = Imputer(strategy="mean")
weight_matrix = imp.fit_transform(weight_matrix)

if(permute_randomly):
	values = []
	for i in range(len(weight_matrix)):
		for j in range(i):
			values.append(weight_matrix[i][j])
	np.random.shuffle(values)
	r = 0
	for i in range(len(weight_matrix)):
		for j in range(i):
			weight_matrix[i][j] = values[r]
			weight_matrix[j][i] = values[r]
			r+=1
if k_nearest:
	weight_matrix = createNearestNeighbor(weight_matrix,k=int(sys.argv[2]))
	if shuffle_k:
		weight_matrix = nx.to_numpy_array(nx.double_edge_swap(nx.from_numpy_array(weight_matrix),nswap=100000,max_tries=100000000))
weight_matrix/=np.sum(weight_matrix)

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

    i = 0
    previous_cost = sess.run(cost)
    sess.run(train_step)
    sess.run(clip)
    initial_difference = previous_cost-sess.run(cost)
 
    while np.isfinite(sess.run(cost)) and (i<=2000 or (previous_cost-sess.run(cost)>.01*initial_difference and i<=10000)):
        previous_cost = sess.run(cost)
        sess.run(train_step)
        sess.run(clip)
        i+=1

    learnt_W = sess.run(W)
    learnt_H = sess.run(H)
        
    error = np.linalg.norm((data-learnt_W.dot(learnt_H))[bool_mask])
    difference = np.linalg.norm((data- learnt_W.dot(learnt_H))[~bool_mask])

    return learnt_H,difference

def run_trials(rank,f,use_random=False,lr=.0001):
    return min([find_results(rank,f,use_random=use_random,lr=lr) for i in range(5)],key=lambda x: x[1])

def find_params():
	possible_lr = [-1,-5,-9]
	possible_lamb = [3,5,7]
	distance_lr = 4
	distance_lamb = 4
	cost_functions.lamb = lamb
	best_score = -1
	for i in range(4):
		results = [[0 for a in range(len(possible_lamb))] for j in range(len(possible_lr))]
		for j in range(len(possible_lr)):
			for k in range(len(possible_lamb)):
				cost_functions.lamb = 10**possible_lamb[k]
				results[j][k] = run_trials(int(sys.argv[1]),frobenius,lr=10**possible_lr[j])[1]
				print(10**possible_lr[j],10**possible_lamb[k])
				print(results)
		best_score = results[0][0]
		best_location = (0,0)
		for j in range(len(results)):
			for k in range(len(results[0])):
				if(results[j][k] < best_score):
					best_score = results[j][k]
					best_location = (j,k)

		distance_lr/=2
		distance_lamb/=2
		best_lr, best_lamb = possible_lr[best_location[0]],possible_lamb[best_location[1]]
		possible_lr = [best_lr-distance_lr,best_lr,best_lr+distance_lr]
		possible_lamb = [best_lamb-distance_lamb,best_lamb,best_lamb+distance_lamb]
		print(possible_lr)
		print(possible_lamb)
	return (10**possible_lr[1],10**possible_lamb[1],best_score)

def find_lr():
        possible_lr = [-1,-5,-9]
        distance_lr = 4
        cost_functions.lamb = 0
        best_score = -1
        for i in range(6):
                results = [0 for j in range(len(possible_lr))]
                for j in range(len(possible_lr)):
                    results[j] = run_trials(int(sys.argv[1]),frobenius,lr=10**possible_lr[j])[1]    
                    print(10**possible_lr[j])
                    print(results)
                best_score = results[0]
                best_location = 0
                for j in range(len(results)):
                    if(results[j] < best_score):
                        best_score = results[j]
                        best_location = j

                distance_lr/=2
                best_lr = possible_lr[best_location] 
                possible_lr = [best_lr-distance_lr,best_lr,best_lr+distance_lr]
                print(possible_lr)

        return (10**possible_lr[1],best_score)

#finding learning rate
t = time.time()
lr,score = find_lr()
print(" ".join(sys.argv[1:]))
print("Score:",score)
print(lr,0)
print("Time",time.time()-t)
