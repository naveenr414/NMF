import numpy as np
from scipy import spatial

class Graph:
    def __init__(self,numNodes):
        self.V = numNodes
        self.adjacency = np.zeros((numNodes,numNodes))

    def addEdge(self,n1,n2,value=1):
        self.adjacency[n1,n2] = value
        self.adjacency[n2,n1] = value

    def calcLaplacian(self):
        self.degree = np.zeros((self.V,self.V))
        for i in range(self.V):
            self.degree[i,i] = np.count_nonzero(self.adjacency[i])
            if(self.adjacency[i][i]):
                self.degree[i,i]+=1
                
        self.laplacian = self.degree-self.adjacency
        return self.laplacian

    def calcDistance(self,W):
        return np.trace(W.T.dot(self.calcLaplacian()).dot(W))

    def __str__(self):
        return str(self.adjacency)

def calcGradient(W,G):
    L = G.calcLaplacian()
    eta = -.01
    return eta*(W.T.dot(L.T) + L.T.dot(W))

def normalize(W):
    for j in range(W.shape[0]):
        W[j] = np.divide(W[j],np.sum(W[j]))

    return W

G = Graph(6)
G.addEdge(0,1,10000)

W = normalize(np.random.rand(6,6))

Wp = Graph(6)
for i in range(6):
    for j in range(6):
        if(i!=j):
            Wp.addEdge(i,j,1 - spatial.distance.cosine(W[i],W[j]))
print(G.calcDistance(W))

for i in range(100):
    W+=calcGradient(W,G)
    W = normalize(W)

Wp = Graph(6)
for i in range(6):
    for j in range(6):
        if(i!=j):
            Wp.addEdge(i,j,1 - spatial.distance.cosine(W[i],W[j]))

print(G.calcDistance(W))
