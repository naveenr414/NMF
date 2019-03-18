from imageEditing import getMatrix
import numpy as np
from scipy import linalg
from pynrnmf import NRNMF
from scipy.optimize import linear_sum_assignment
import time
from util import createNearestNeighbor

def diracDelta(a,b):
    return int(a==b)

def getPredictions(objects,angles):
    """Get the predicted factorization"""
    
    V = getMatrix(objects,angles)
    graph = createNearestNeighbor(V,k=9)

    model = NRNMF(k=objects,W=graph,alpha=100,init='random',n_inits=1,max_iter=50000,n_jobs=1)
    H, W = model.fit_transform(V.T)
    H = H.T

    predictedAnswers = []
    for row in W:
        predictedAnswers.append(np.argmax(row))
        
    return predictedAnswers

def getCorrect(objects,angles):
    """What's the correct classification for object/angle pair"""
    correctAnswers = []
    for obj in range(objects):
        for angle in range(angles):
            correctAnswers.append(obj)
    return correctAnswers

def mapPredicitions(objects,correctAnswers,predictedAnswers):
    """Use Hungarian Algorithm to maximize mapping between predicitions, correct"""
    hungarianMatrix = np.zeros((objects,objects))
    for i in range(len(correctAnswers)):
        hungarianMatrix[predictedAnswers[i]][correctAnswers[i]]+=1
    row,col = linear_sum_assignment(-1*hungarianMatrix)
    AC = sum(map(lambda x: diracDelta(correctAnswers[x],col[predictedAnswers[x]]),range(len(correctAnswers))))
    AC/=len(correctAnswers)

    return AC


start = time.time()

objects = 20
angles = 72
correctAnswers = getCorrect(objects,angles)
predictedAnswers = getPredictions(objects,angles)
AC = mapPredicitions(objects,correctAnswers,predictedAnswers)
print(AC)

print(time.time()-start)
