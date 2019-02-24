import tensorflow as tf
import numpy as np
import pandas as pd
from copy import deepcopy
#np.random.seed(1000)

def create_random(m,n,k):
    W = np.random.random((m,k))
    H = np.random.random((k,n))
    V = W.dot(H).astype(np.float32)

    return V

def create_mask(m,n,percent_taken=0.5):
    all_cells = [(i,j) for i in range(m) for j in range(n)]
    np.random.shuffle(all_cells)
    masked_cells = all_cells[:int(percent_taken*m*n)]
    return masked_cells

def mask_random(V,masked_cells):
    m,n = V.shape

    V = deepcopy(V)

    for i,j in masked_cells:
        V[i,j] = np.NAN

    return V

def run(np_mask,A_df_masked,rank):
    m,n = A_df_masked.shape
    
    temp_H = np.random.randn(k, n).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.random.randn(m, k).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    tf_mask = tf.Variable(np_mask.values)
    A = tf.constant(A_df_masked)   
    H =  tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))
    lr = 0.001
    # Number of steps
    steps = 1000
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()

    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)

    steps = 1000
    sess = tf.Session()
    sess.run(init)
    for i in range(steps):
        sess.run(train_step)
        sess.run(clip)

    learnt_W = sess.run(W)
    learnt_H = sess.run(H)
    learnt_WH = learnt_W.dot(learnt_H)

    return learnt_WH


m = 50
n = 40
k = 25
percent_taken = 0.5

A_orig = create_random(m,n,k)
error_list = {}

for i in range(5):
    masked_cells = create_mask(m,n)
    A_df_masked = mask_random(A_orig,masked_cells)
    np_mask = pd.DataFrame(A_df_masked).notnull()

    for rank in range(max(k-10,2),k+10,2):
        learnt_WH = run(np_mask,A_df_masked,rank)
        
        diff = np.abs(learnt_WH-A_orig)
        error = sum([diff[i,j] for i,j in masked_cells])
        if rank not in error_list:
            error_list[rank] = 0
        error_list[rank]+=error

sorted_errors = list(error_list.items())
sorted_errors = sorted(sorted_errors,key=lambda x: x[1])
for rank,error in sorted_errors:
    print(rank,error)
