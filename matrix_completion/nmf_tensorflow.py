import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
import util
import cost_functions
import time
import networkx as nx

np.set_printoptions(suppress=True)

TARGET_DIFFERENCE = 0.1
max_iterations = 100000

def create_mask(matrix,hidden_pairs):
    matrix_df = pd.DataFrame(matrix)

    matrix_df_masked = matrix_df.copy()

    for (i, j) in hidden_pairs:
        matrix_df_masked.iloc[i, j] = np.NAN

    return matrix_df_masked

def init_W_H(shape, rank):
    temp_W = np.abs(np.random.randn(shape[0], rank)).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())
    W = tf.Variable(temp_W)

    temp_H = np.abs(np.random.randn(rank, shape[1])).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())
    H = tf.Variable(temp_H)

    return (W, H)

def get_clip(W, H):
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)
    return clip

def find_results(data,weight_matrix,params):
    """Run the optimization function on data with a weight matrix
    The params are
        lambda
        rank
        lr
        hidden_pairs (array of tuples)
    """
    
    data = data.astype(np.float32)
    weight_matrix = weight_matrix.astype(np.float32)
    
    rank = params['rank']
    lamb = params['lambda']
    lr = params['lr']
    hidden_pairs = params['hidden_pairs']
    cost_functions.lamb = lamb

    f = cost_functions.frobenius 
    V_masked = create_mask(data,hidden_pairs)
    bool_mask = V_masked.notnull().values
    tf_mask = tf.Variable(bool_mask)
    
    V = tf.constant(V_masked.values)
    laplacian_matrix = laplacian(weight_matrix).astype(np.float32)
    W, H = init_W_H(V.shape, rank=rank)
    WH = tf.matmul(W, H)
    L = tf.constant(laplacian_matrix)
    WTLW = tf.matmul(tf.matmul(tf.transpose(W), L), W)

    cost = f(V, tf_mask, WH, WTLW)
    train_step = tf.train.ProximalGradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()
    clip = get_clip(W, H)

    sess = tf.Session()
    sess.run(init)

    previous_cost = sess.run(cost)
    sess.run(train_step)
    sess.run(clip)
    initial_difference = previous_cost - sess.run(cost)

    matrix_errors = []
    graph_errors = []
    imputation_error = []

    learnt_W = sess.run(W).astype(np.float32)
    learnt_H = sess.run(H).astype(np.float32)
    imputation_norm = np.linalg.norm((data - learnt_W.dot(learnt_H))[~bool_mask])
    
    i = 0
    while np.isfinite(sess.run(cost)) and previous_cost-sess.run(cost) > TARGET_DIFFERENCE * initial_difference and i<=max_iterations:
        previous_cost = sess.run(cost)
        sess.run(train_step)
        sess.run(clip)
        matrix_errors.append(sess.run(cost_functions.matrix_cost))
        graph_errors.append(sess.run(cost_functions.graph_cost))
        i+=1

        learnt_W = sess.run(W).astype(np.float32)
        learnt_H = sess.run(H).astype(np.float32)

        imputation_norm = np.linalg.norm((data - learnt_W.dot(learnt_H))[~bool_mask])
        imputation_error.append(imputation_norm)

    return {'imputation_error':imputation_norm,'W':sess.run(W),'H':sess.run(H),
            'graph_error':graph_errors,'matrix_error':matrix_errors,'imputation_error_list':imputation_error}
