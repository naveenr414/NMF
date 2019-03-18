#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import spatial
import scipy.optimize

signatures_file = 'signatures.npy'
data_file = 'data.npy'

def signature_similarity(my_signatures, real_signatures):
    num_signatures = len(real_signatures)
    similarities = np.zeros((len(my_signatures), len(real_signatures)))
    for i in range(len(my_signatures)):
        for j in range(len(real_signatures)):
            a = np.array(my_signatures[i])
            b = np.array(real_signatures[j])
            similarities[i][j] = spatial.distance.cosine(a, b)
    avg = 0
    best = scipy.optimize.linear_sum_assignment(similarities)
    for i in range(len(best[0])):
        avg += similarities[best[0][i]][best[1][i]]
    avg /= len(best[0])
    return avg

def find_results(rank,HIDDEN_PERCENT = 0.1,lr = 0.001, steps = 1000):
    real_signatures = np.load(signatures_file)


    A_orig = data = np.load(data_file).astype(np.float32)
    all_pairs = [(i, j) for i in range(A_orig.shape[0]) for j in
                 range(A_orig.shape[1])]
    A_orig_df = pd.DataFrame(A_orig)

    
    np.random.shuffle(all_pairs)
    hidden_pairs = all_pairs[:int(round(HIDDEN_PERCENT
                                  * len(all_pairs)))]

    A_df_masked = A_orig_df.copy()

    for (i, j) in hidden_pairs:
        A_df_masked.iloc[i, j] = np.NAN

    np_mask = A_df_masked.notnull()
    tf_mask = tf.Variable(np_mask.values)
    A = tf.constant(A_df_masked.values)
    shape = A_df_masked.values.shape


    temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.random.randn(shape[0], rank).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H = tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask)
                         - tf.boolean_mask(WH, tf_mask), 2))
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()

    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            sess.run(train_step)
            sess.run(clip)
        learnt_W = sess.run(W)
        learnt_H = sess.run(H)

    difference = np.linalg.norm((A_orig
                                - learnt_W.dot(learnt_H))[~np_mask])
    results = {}
    results['rank'] = rank
    results['norm'] = difference
    results['signature'] = signature_similarity(learnt_H,real_signatures)
    return results

def find_results_no_mask(rank,HIDDEN_PERCENT = 0.1,lr = 0.001, steps = 1000):
    real_signatures = np.load(signatures_file)

    A_orig = data = np.load(data_file).astype(np.float32)
    A_orig_df = pd.DataFrame(A_orig)

    A_df_masked = A_orig_df.copy()

    np_mask = A_df_masked.notnull()
    tf_mask = tf.Variable(np_mask.values)
    A = tf.constant(A_df_masked.values)
    shape = A_df_masked.values.shape


    temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.random.randn(shape[0], rank).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H = tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask)
                         - tf.boolean_mask(WH, tf_mask), 2))
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()

    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            sess.run(train_step)
            sess.run(clip)
        learnt_W = sess.run(W)
        learnt_H = sess.run(H)

    difference = np.linalg.norm((A_orig
                                - learnt_W.dot(learnt_H)))
    results = {}
    results['rank'] = rank
    results['norm'] = difference
    results['signature'] = signature_similarity(learnt_H,real_signatures)
    return results

if __name__ == '__main__':
    for rank in range(2,10):
        print(find_results_no_mask(rank))
