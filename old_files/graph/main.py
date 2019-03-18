from examples import *
from math import e
import matplotlib.pyplot as plt
import util
from pynrnmf import NRNMF
import numpy as np
from copy import deepcopy

def createNearestNeighbor(mat,k=1):
    numRows = mat.shape[0]
    graph = np.zeros((numRows,numRows))
    dist = np.zeros((numRows,numRows))
    for row in range(numRows):
        for otherRow in range(numRows):
            if(otherRow==row):
                dist[row][otherRow] = 10000000
            else:
                dist[row][otherRow] = np.linalg.norm(mat[row]-mat[otherRow])

    for row in range(numRows):
        smallestNums = np.argpartition(dist[row],k)[:k]
        for i in smallestNums:
            graph[row][i] = 1
            graph[i][row] = 1

    return graph

def normalize(row):
    s=sum(row)
    for i in range(len(row)):
        row[i]/=s
    return row

def runIteration(graph,alpha,num=0):
    V = deepcopy(allData)
    k=5
    model = NRNMF(k=k,W=graph,alpha=0*10**alpha,init='random',n_inits=4,max_iter=250)
    H, W = model.fit_transform(V.T)
    H = H.T

    """
    print("\t Mean \t Std")

    for rowNum,row in enumerate(H):
        row = normalize(row)
        mean = round(np.mean(row),2)
        std = round(np.std(row),2)
        print(rowNum,"\t",mean,"\t",std)
    """

    plt.bar(range(H.shape[1]),H[num])
    plt.show()

cancerList = ["bladder"]
for i in range(len(cancerList)):
    print(100*i,cancerList[i])

rows = 100

allData = np.empty((0,0))

for i in cancerList:
    with open("graph/alexandrov_data/"+i+"_exome.txt") as f:
        cancerData = f.read().split("\n")[1:-1]
        for i in range(0,len(cancerData)):
            cancerData[i] = cancerData[i].split("\t")[1:]
            cancerData[i] = list(map(int,cancerData[i]))

    cancerData = np.array(cancerData).T

    cancerData = cancerData[np.random.choice(cancerData.shape[0], min(rows,cancerData.shape[0]), replace=False)]

    if(not allData.shape[0]):
        allData = cancerData
    else:
        allData = np.append(allData,cancerData,axis=0)


V = deepcopy(allData)
k = 5
graph = createNearestNeighbor(V,k=2)

for i in range(5):
    print("alpha = 0, row =",i)
    runIteration(graph,0,num=3)

for k in range(1,3):
    graph = createNearestNeighbor(V,k=k)
    for alpha in range(0,5):
        print("k =",k,", alpha = ",alpha)
        runIteration(graph,10**alpha,num=3)






