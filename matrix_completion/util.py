import numpy as np
import scipy

epsilon = .8

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
                    dist[row][otherRow] = 1-scipy.spatial.distance.cosine(mat[row],mat[otherRow])#np.linalg.norm(mat[row]-mat[otherRow])
                elif(metric=="Norm"):
                    dist[row][otherRow] = np.linalg.norm(mat[row]-mat[otherRow])
    
    for row in range(numRows):
        smallestNums = np.argpartition(dist[row],k)[:k]
        for i in smallestNums:
            graph[row][i] = 1
            graph[i][row] = 1

    return graph

def createNearestNeighborEpsilon(mat,epsilon=0.3,metric="Cosine"):
    numRows = mat.shape[0]
    graph = np.zeros((numRows,numRows))
    dist = np.zeros((numRows,numRows))
    for row in range(numRows):
        for otherRow in range(numRows):
            if(otherRow==row):
                dist[row][otherRow] = 10000000
            else:
                if(metric=="Cosine"):
                    dist[row][otherRow] = 1-scipy.spatial.distance.cosine(mat[row],mat[otherRow])#np.linalg.norm(mat[row]-mat[otherRow])
                elif(metric=="Norm"):
                    dist[row][otherRow] = np.linalg.norm(mat[row]-mat[otherRow])
    
    for row in range(numRows):
        for otherRow in range(numRows):
            if(dist[row][otherRow]<=epsilon):
                graph[row][otherRow] = 1
                graph[otherRow][row] = 1

    return graph

def createAgeGraph(age_file,data_file):
    samples = open(data_file).read().split("\n")[0].split("\t")[1:]
    for i in range(len(samples)):
        samples[i] = samples[i][:-1]
    age_dict = {}
    ages = open(age_file).read().split("\n")
    ages = [i.split("\t") for i in ages]
    for i in ages:
        if(i[1] == "over_80"):
            age_dict[i[0]] = 80
        else:
            age_dict[i[0]] = int(i[1])

    data = open(data_file).read().split("\n")[1:-1]
    for i in range(len(data)):
        data[i] = data[i].split("\t")[1:]
        data[i] = list(map(int,data[i]))

    data = np.array(data).T
    i = 0
    while(i<len(samples)):
        if samples[i] not in age_dict:
            del samples[i]
            data = np.delete(data,i,axis=0)
        else:
            i+=1

    similarities = np.zeros((data.shape[0],data.shape[0]))
    for i in range(len(samples)):
        for j in range(len(samples)):
            similarities[i][j] = 1-float(abs(age_dict[samples[i]]-age_dict[samples[j]]))/float(80)

    return data.astype(np.float32),similarities.astype(np.float32)

def normalize(m):
    for i in range(len(m)):
        m[i] = np.divide(m[i],np.sum(m[i]))

    return m
    
