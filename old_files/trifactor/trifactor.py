import numpy as np
import scipy 

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

def trifactor(X,sigma,k,iterations=100):
    Z = X
    H = np.random.rand(X.shape[0],k)
    S = np.random.rand(k,k)
    Y = H.dot(S).dot(H.T)

    for i in range(iterations):
        if(i%10 == 0):
            print(i)
        
        topH = Z.dot(H.dot(S))
        bottomH = H.dot(S).dot(H.T).dot(H).dot(S)

        topS = H.T.dot(Z.dot(H))
        bottomS = H.T.dot(H.dot(S).dot(H.T)).dot(H)

        for l in range(H.shape[0]):
            for m in range(H.shape[1]):
                if(bottomH[l,m] != 0):
                    H[l,m]*=(topH[l,m]/bottomH[l,m])**(1/4)
            
        for l in range(S.shape[0]):
            for m in range(S.shape[1]):
                if(bottomS[l,m] != 0):
                    S[l,m]*=(topS[l,m]/bottomS[l,m])
        
        Y = H.dot(S).dot(H.T)
        Z = X+Y

        for j in sigma:
            Z[j] = X[j]

    return Y

M = np.array([[0,1,0],[0,0,1],[0,0,0]])
sigma = [(0,1),(1,2)]
