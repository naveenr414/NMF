import numpy as np
import scipy

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
