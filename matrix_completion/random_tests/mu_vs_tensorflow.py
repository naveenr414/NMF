from create_data import create_data
import numpy as np
import pandas as pd
import tensorflow as tf 

def f(V,W,H):
    VP = W.dot(H)
    F = sum([abs(V.item(i)*np.log(VP.item(i)) - VP.item(i)) for i in range(V.size)])
    return F

def multiplicativeUpdate(V,q):
    p = V.shape[0]
    n = V.shape[1]
    
    W = makeRandom(p,q)*np.linalg.norm(V,'fro')
    H = makeRandom(q,n)*np.linalg.norm(V,'fro')


    previousF = -1

    while(previousF==-1 or previousF-f(V,W,H)>.001):
        previousF = f(V,W,H)
        for a in range(q):
            for b in range(n):
                H[a,b] = H[a,b]*(sum([W[i,a]*V[i,b]/
                            sum([W[i,k]*H[k,b] for k in range(q)]) for i in range(p)]))/sum([W[i,a] for i in range(p)])
                            
            for c in range(p):
                W[c,a] = W[c,a]*(sum([H[a,j]*V[c,j]/
                                 sum([W[c,k]*H[k,j] for k in range(q)]) for j in range(n)]))/sum([H[a,j] for j in range(n)])
                W[c,a] = W[c,a]/sum([W[j,a] for j in range(p)])
        numTrials = 0

    return (W,H)
def makeRandom(m,n):
    return np.random.rand(m,n)

for i in range(10):
    create_data()
    data_file = 'data.npy'
    lr = 0.001
    steps = 1000
    A_orig = np.load(data_file).astype(np.float32)
    rank = 5
    W,H = multiplicativeUpdate(A_orig,rank)
    print("MU",np.linalg.norm(A_orig-W.dot(H)))

    A_orig_df = pd.DataFrame(A_orig)
    A = tf.constant(A_orig_df.values)

    shape = A_orig_df.values.shape

    temp_H = np.abs(np.random.randn(rank, shape[1])).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.abs(np.random.randn(shape[0], rank)).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H = tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    matrix_cost = tf.reduce_sum(tf.pow(A- WH, 2))
    cost = matrix_cost

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

    print("TF",np.linalg.norm(A_orig-learnt_W.dot(learnt_H)))
