import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from util import makeRandom, readTSV

np.set_printoptions(suppress=True)

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

def nonSmooth(V,q,theta):
    p = V.shape[0]
    n = V.shape[1]

    W = makeRandom(V.shape[0],q)*np.linalg.norm(V,'fro')
    H = makeRandom(q,V.shape[1])*np.linalg.norm(V,'fro')

    S = (1-theta)*np.identity(q) + theta/q * np.array([1 for i in range(q)])*np.transpose(np.array([1 for i in range(q)]))

    previousF = -1

    while(previousF==-1 or previousF-f(V,W,H)>.001):
        previousF = f(V,W,H)
        for a in range(q):
            WS = W.dot(S)
            for b in range(n):
                H[a,b] = H[a,b]*(sum([WS[i,a]*V[i,b]/
                            sum([WS[i,k]*H[k,b] for k in range(q)]) for i in range(p)]))/sum([WS[i,a] for i in range(p)])

            SH = S.dot(H)
                 
            for c in range(p):
                W[c,a] = W[c,a]*(sum([SH[a,j]*V[c,j]/
                                 sum([W[c,k]*SH[k,j] for k in range(q)]) for j in range(n)]))/sum([SH[a,j] for j in range(n)])
                W[c,a] = W[c,a]/sum([W[j,a] for j in range(p)])

    return (W,H)


