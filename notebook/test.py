 
import glob
import random
import numpy as np
import scipy

SAMPLES = 100
CANCER_TYPES = 1 # How many of the types of cancer should we use
normalize = False
outliers = True
restrictSamples = False # Remove cancer types with less than SAMPLES samples

def generateColors(n):
    """ Generates n random colors in hex """
    
    colorList = []
    possibleValues = list(range(10))+["A","B","C","D","E","F"]
    
    for i in range(n):
        tempColor = "#"
        for j in range(6):
            tempColor+=str(possibleValues[random.randint(0,len(possibleValues)-1)])
        colorList.append(tempColor)
    
    return colorList

def toArray(fileName):
    """Converts exome data to an NP array """
    
    cancerData = ""
    with open(fileName) as f:
        cancerData = f.read().split("\n")[1:-1]
        for i in range(0,len(cancerData)):
            cancerData[i] = cancerData[i].split("\t")[1:]
            cancerData[i] = list(map(int,cancerData[i]))

    cancerData = np.array(cancerData,dtype=np.float).T
    return cancerData

def normalizeRows(mat):
    """ Normalized the rows of the matrix mat """
    
    mat = deepcopy(mat)
    
    for i in range(mat.shape[0]):
        s = sum(mat[i])
        for j in range(mat.shape[1]):
            num = float(mat[i][j])/float(s)
            mat[i][j] = float(mat[i][j])
            mat[i][j]/=float(s)

            mat[i][j] = num

    return mat
#This sections inits all the cancer types 
alexandrovFiles = ['alexandrov_data/breast_exome.txt']#glob.glob("alexandrov_data/*.txt")

startingPositions = []
allData = np.empty((0,0),dtype=np.float)

if(CANCER_TYPES<len(alexandrovFiles)):
    alexandrovFiles = np.random.choice(alexandrovFiles,CANCER_TYPES,replace=False)
CANCER_TYPES = len(alexandrovFiles)

cancerNames = [i.split("\\")[-1].replace("_","").split("exome")[0] for i in alexandrovFiles]

colors = generateColors(len(cancerNames))
unusedCancers = []

#Combines the data from each of the samples
for i,file in enumerate(alexandrovFiles):
    cancerData = toArray(file)
    sampleRows = cancerData.shape[0]    
    selectedSamples = np.random.choice(sampleRows,min(sampleRows,SAMPLES),replace=False)

    if(sampleRows>=SAMPLES or True):
        if(len(startingPositions)==0):
            startingPositions.append(0)

        startingPositions.append(startingPositions[-1] + min(sampleRows,SAMPLES))


        cancerData = cancerData[selectedSamples]
        
        if(allData.size == 0): 
            allData = cancerData
        else:
            allData = np.append(allData,cancerData,axis=0)
    else:
        unusedCancers.append(i)

cancerNames = [i for j,i in enumerate(cancerNames) if j not in unusedCancers]
CANCER_TYPES = len(cancerNames)


if(normalize):
    allData = normalizeRows(allData)

np.savetxt("allData.csv", allData, delimiter=",")

def findMutationOrder(fileName):
    f = open(fileName).read().split("\n")
    mutationList = []
    for i in f:
        i = i.split(" ")
        for j in i:
            if("[" in j):
                mutationList.append(j.strip("\t").split("\t")[0])
    return mutationList

import scipy.spatial.distance
import warnings
warnings.filterwarnings("ignore")

def distanceGraph(mat,metric="Cosine"):
    numRows = mat.shape[0]
    graph = np.zeros((numRows,numRows))
    dist = np.zeros((numRows,numRows))
    for row in range(numRows):
        for otherRow in range(numRows):
            if(otherRow==row):
                dist[row][otherRow] = 10000000
            else:
                if(metric=="Cosine"):
                    dist[row][otherRow] = scipy.spatial.distance.cosine(mat[row],mat[otherRow])
                elif(metric=="Norm"):
                    dist[row][otherRow] = np.linalg.norm(mat[row]-mat[otherRow])
    return dist

def createNearestNeighborExponential(mat, lamb=0.3,metric="Cosine"):
    dist = distanceGraph(mat,metric=metric)
    numRows = mat.shape[0]
    graph = np.zeros((numRows,numRows))  

    for row in range(numRows):
        epsilon = np.random.exponential(lamb)
        for otherRow in range(row+1):
            if(dist[row][otherRow]<=epsilon):
                graph[row][otherRow] = 1
                graph[otherRow][row] = 1
            else:
                graph[row][otherRow] = 0
                graph[otherRow][row] = 0

    return graph


def createNearestNeighborEpsilon(mat,epsilon=0.3,metric="Cosine"):
    dist = distanceGraph(mat,metric=metric)
    numRows = mat.shape[0]
    graph = np.zeros((numRows,numRows))  

    for row in range(numRows):
        for otherRow in range(numRows):
            if(dist[row][otherRow]<=epsilon):
                graph[row][otherRow] = 1
                graph[otherRow][row] = 1
            else:
                graph[row][otherRow] = 0
                graph[otherRow][row] = 0

    return graph

def createNearestNeighbor(mat,k=1,metric="Norm"):
    numRows = mat.shape[0]
    graph = np.zeros((numRows,numRows))
    dist = np.zeros((numRows,numRows))
    for row in range(numRows):
        for otherRow in range(numRows):
            if(otherRow==row):
                dist[row][otherRow] = 10000000
            else:
                if(metric=="Cosine"):
                    dist[row][otherRow] = scipy.spatial.distance.cosine(mat[row],mat[otherRow])#np.linalg.norm(mat[row]-mat[otherRow])
                elif(metric=="Norm"):
                    dist[row][otherRow] = np.linalg.norm(mat[row]-mat[otherRow])
    
    for row in range(numRows):
        smallestNums = np.argpartition(dist[row],k)[:k]
        for i in smallestNums:
            graph[row][i] = 1
            graph[i][row] = 1

    return graph

from sklearn.utils import check_random_state, check_array
from numpy.linalg import norm
import scipy.sparse as sp

def GNMF(X,L,lambd=0,n_components=None,tol=1e-4,max_iter=100,verbose=False):
        n_samples, n_features = X.shape

        if not n_components:
            n_components = n_features
        else:
            n_components = n_components

        #W, H = NBS_init(X,n_components)
        W = np.random.normal(0,1,(n_samples,n_components))**2
        H = np.random.normal(0,1,(n_components,n_features))**2

        reconstruction_err_ = norm(X - np.dot(W, H))
        eps=1e-4#spacing(1) #10e-14
        Lp = (abs(L)+L)/2
        Lm = (abs(L)-L)/2

        for n_iter in range(1, max_iter + 1):
            h1=lambd*np.dot(H,Lm)+np.dot(W.T,(X+eps)/(np.dot(W,H)+eps))
            h2=lambd*np.dot(H,Lp)+np.dot(W.T,np.ones(X.shape))
            H = np.multiply(H,(h1+eps)/(h2+eps))
            H[H<=0]=eps
            H[np.isnan(H)]=eps

            w1=np.dot((X+eps)/(np.dot(W,H)+eps),H.T)
            w2=np.dot(np.ones(X.shape),H.T)
            W = np.multiply(W,(w1+eps)/(w2+eps))
            W[W<=0]=eps
            W[np.isnan(W)]=eps

            if not sp.issparse(X):
                if reconstruction_err_ > norm(X - np.dot(W, H)):
                    H=(1-eps)*H+eps*np.random.normal(0,1,(n_components,n_features))**2
                    W=(1-eps)*W+eps*np.random.normal(0,1,(n_samples,n_components))**2
                reconstruction_err_ = norm(X - np.dot(W, H))
            else:
                norm2X = np.sum(X.data ** 2)  # Ok because X is CSR
                normWHT = np.trace(np.dot(np.dot(H.T, np.dot(W.T, W)), H))
                cross_prod = np.trace(np.dot((X * H.T).T, W))
                reconstruction_err_ = np.sqrt(norm2X + normWHT - 2. * cross_prod)

        return np.squeeze(np.asarray(W)), np.squeeze(np.asarray(H)).T, reconstruction_err_

from scipy.sparse import csgraph
import random
from copy import deepcopy

def D(X,lamb):
    U,s,V = np.linalg.svd(X)
    for i in range(len(s)):
        s[i] = max(0,s[i]-lamb)
    
    sig = np.zeros((U.shape[1],V.shape[0]))
    for i in range(len(s)):
        sig[i][i] = s[i]
    
    return U.dot(sig).dot(V)

def grad(X,Q,Y,L,lambOne,sigma,listOf):
    d,n = X.shape
    r = np.zeros((d,n))
    proj = Q.T.dot(X)
    diff = 2*(proj-Y)
    
    for i,j in sigma:
        r+=diff[i][j]*listOf[i][j]
        
    r+=2*lambOne*Q.dot(Q.T).dot(X).dot(L)
    return r

def g(X,Q,Y,L,lambOne,sigma):
    I = np.zeros(Y.shape)
    for i,j in sigma:
        I[i,j] = 1
    m = np.linalg.norm(I - Q.T.dot(X),ord='fro')**2
    m+=lambOne*np.trace(Q.T.dot(X).dot(L).dot(X.T).dot(Q))
    return m
        
def h(X,Q,Y,L,lambOne,lambTwo,sigma):
    return g(X,Q,Y,L,lambOne,sigma) + lambTwo*np.linalg.norm(X,ord='nuc')

def Gp(X,Q,Y,L,lambOne,lambTwo,Xk1,rhok1,sigma,gr):
    m = np.linalg.norm(X-(Xk1 - 1/(rhok1)*gr),ord='fro')**2
    m*=rhok1/2
    m+=lambTwo*np.linalg.norm(X,ord='nuc')
    return m

def matrixComplete(W,k,lambOne,lambTwo,sigma,rhoK1=1000,gamma=100):    
    np.random.seed()
    percent = .05
    Q = GNMF(allData.T,W,0,n_components=k,tol=1e-4,max_iter=100,verbose=False)[0]
    Q = Q.T
    sigma = [(j,i) for i,j in sigma]
    L = csgraph.laplacian(W, normed=False)
    Y = deepcopy(allData).T
    al = [(i,j) for i in range(Y.shape[0]) for j in range(Y.shape[1])]
    notSigma = list(set(al)-set(sigma))

    eList = [np.zeros((allData.shape[0],1)) for i in range(allData.shape[0])]
    for i in range(len(eList)):
        eList[i][i] = 1
    

    QList = [Q[:,[i]] for i in range(Q.shape[1])]
    listOf = [[(QList[i]).dot(eList[j].T) for j in range(len(eList))] for i in range(len(QList))]

    for i,j in notSigma:
        Y[i][j] = -1
    
    
    rhoK = rhoK1
    Xk1 = np.random.rand(Q.shape[0],allData.shape[0])

    for i in range(10):
        gr = grad(Xk1,Q,Y,L,lambOne,sigma,listOf)
        Xk = D(Xk1 - 1/(rhoK1)*gr,lambTwo/rhoK1)
        rhoK = rhoK1
        l = h(Xk,Q,Y,L,lambOne,lambTwo,sigma)
        r = Gp(Xk,Q,Y,L,lambOne,lambTwo,Xk1,rhoK,sigma,gr)
        while(l>r):
            rhoK*=gamma
            r = Gp(Xk,Q,Y,L,lambOne,lambTwo,Xk1,rhoK,sigma,gr)

        Xk1 = deepcopy(Xk)
        rhoK1 = deepcopy(rhoK)

    return Q,Xk

import random

k = 10
threshold = 0.95
W = createNearestNeighborEpsilon(allData)
sigma = [(i,j) for i in range(allData.shape[0]) for j in range(allData.shape[1]) if random.random()<threshold]

def gridSearch(W,k,lamb,rhoK1,gamma):
    Q, Xk = matrixComplete(W,k,lamb,0,sigma,rhoK1=10**rhoK1,gamma=10**gamma)
    mat = allData-(Q.T.dot(Xk)).T
    
    reconstructionError = np.linalg.norm(mat)**2-sum([mat[j,k]**2 for j,k in sigma])
    reconstructionError = np.sqrt(reconstructionError)

    return (reconstructionError,(lamb,rhoK1,gamma))

import time
import multiprocessing as mp

a = time.time()

k=10

lambList = [0.1]
rhoCenters = [2 for i in range(len(lambList))]
gammaCenters = [2 for i in range(len(lambList))]
scores = [10000000 for i in range(len(lambList))]
std = 1

for iteration in range(1):
    for i in range(len(lambList)):
        lamb = lambList[i]
        rhoKList = [rhoCenters[i]+std*t for t in range(-1,2)]
        gammaList = [gammaCenters[i]+std*t for t in range(-1,2)]
        for rhoK1 in rhoKList:
            for gamma in gammaList:
                print(gridSearch(W,k,lamb,rhoK1,gamma))




"""
    print(results)
    std/=1.5

    for i in results:
        reconstructionError = i[0]
        lamb = i[1][0]
        lambNumber = lambList.index(lamb)
        rho = i[1][1]
        gamma = i[1][2]
        if(reconstructionError<scores[lambNumber]):
            scores[lambNumber] = reconstructionError
            rhoCenters[lambNumber] = rho
            gammaCenters[lambNumber] = gamma
            
    #for i in range(len(lambList)):
    #    print(lambList[i],scores[i],rhoCenters[i],gammaCenters[i])
        
"""
print("Time taken parallel",int(time.time()-a))


