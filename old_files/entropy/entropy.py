import numpy as np
from copy import deepcopy
from math import log, e
import util

def normalize(H):
    return H/np.sum(H)

def entropy(H):
    H = normalize(H)
    H = np.array([-np.log(i)*i for i in H])
    return np.sum(H)

def gradEntropy(H):
    H = normalize(H)
    try:
        H = -np.log(H)
    except:
        print(H)
    H = np.add(H,1)
    return H

def gradW(V,W,H,epsilon=0,distro=-1):    
    a = (-V/(W.dot(H))).dot(H.T)+np.ones(V.shape).dot(H.T)

    entropyGradient = np.ones(W.shape)
    #Normalize all of W
    if(distro==0):
        entropyGradient = gradEntropy(W)
    elif(distro==1):
        for i in range(W.shape[1]):
            entropyGradient[:,i] = gradEntropy(W[:,i])
    a=np.subtract(a,np.multiply(epsilon,entropyGradient))

    return a

def gradH(V,W,H,epsilon=0.01,distro=-1):
    a = -W.T.dot(V/W.dot(H)) + W.T.dot(np.ones(V.shape))
    
    entropyGradient = np.ones(H.shape)
    #Normalize all of H
    if(distro==0):
        entropyGradient = gradEntropy(H)
        
    #Normalize the rows of H, then add up the entropies
    elif(distro==1):
        for i in range(H.shape[0]):
            entropyGradient[i] = gradEntropy(H[i])

    a-=epsilon*entropyGradient
    return a

def iterate(V,W,H,method=0,epsilonH=0,epsilonW=0,distributionH=0,distributionW=0,nuW=1,nuH=1):
    SMALL = .0000001


    #The original W and H
    WC = deepcopy(W)
    HC = deepcopy(H)

    #If we haven't already defined a nu
    nuW= np.multiply(nuW,W/(W.dot(H).dot(H.T))/10)
    nuH= np.multiply(nuH,H/(W.T.dot(W).dot(H))/10)

    #Change all negatives to SMALL (basically 0)
    if(method == 0):
        H[H<=0] = SMALL
        W[W<=0] = SMALL

    #Apply gradient descent
    
    W=np.subtract(W,np.multiply(nuW,gradW(V,WC,HC,epsilonW,distributionW)))
    H=np.subtract(H,np.multiply(nuH,gradH(V,WC,HC,epsilonH,distributionH)))

    #Change all negatives to SMALL (basically 0)
    if(method == 0):
        H[H<=0] = SMALL
        W[W<=0] = SMALL

    #Shift by the smallest element
    elif(method == 1):
        H = np.add(H,max(SMALL,-np.amin(H)+SMALL))
        W = np.add(W,max(SMALL,-np.amin(W)+SMALL))

    return W,H

def runIterations(V,k,numIterations=1000,mu=2.5,sigma=1,epsilon=e**-12,method=1,distribution=0,nuW=1000,nuH=1000):
    W,H = np.zeros((V.shape[0],k)), np.zeros((k,V.shape[1]))
    W = np.random.normal(mu,sigma,W.shape[0]*W.shape[1]).reshape(W.shape)
    H = np.random.normal(mu,sigma,H.shape[0]*H.shape[1]).reshape(H.shape)

    W[W<=0] = .000001
    H[H<=0] = .000001

    for i in range(numIterations):
        W,H = iterate(V,W,H,epsilonW=epsilon,method=method,distributionW=distribution,nuW=nuW,nuH=nuH)   
        isNan = True in np.isnan(W) or True in np.isnan(H) 

        if(isNan):
            W,H = previousW, previousH
            break

        W[W<=0] = .000001
        H[H<=0] = .000001

        previousW, previousH =  W, H
        
    W[W<=0] = .000001
    H[H<=0] = .000001

    return W, H
        

#KL Divergence 
def D(V,W,H):
    tot = 0
    prod = W.dot(H)
    for m in range(V.shape[0]):
        for n in range(V.shape[1]):
            tot+=V[m,n]*log(V[m,n]/(prod[m,n])) - V[m,n] + prod[m,n]

    return tot

#Frobenius Distance
def frobenius(V,W,H):
    K = V-W.dot(H)
    return np.sum(K**2)

V = util.readTSV("data/classdata/mutation-counts.tsv")
V = np.add(V,.000001)
k=5
W,H = np.zeros((V.shape[0],k)), np.zeros((k,V.shape[1]))
W = np.random.poisson(1,W.shape[0]*W.shape[1]).reshape(W.shape)
H = np.random.poisson(1,H.shape[0]*H.shape[1]).reshape(H.shape)
W = np.add(W,.000001)
H = np.add(H,.000001)
iterate(V,W,H)
