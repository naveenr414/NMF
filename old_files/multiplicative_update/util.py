import numpy as np
from scipy.optimize import linear_sum_assignment

def makeRandom(m,n):
    return np.random.rand(m,n)
    

def readTSV(name):
	t = []
	f = open(name).read().split("\n")[1:-1]
	for i in range(len(f)):
		t.append(list(map(int,f[i].split("\t")[1:])))
	return np.array(t)

def cosineSimilarity(a,b):
    top = sum([a.item(i)*b.item(i) for i in range(a.size)])
    bottom = sum([a.item(i)**2 for i in range(a.size)])**.5
    bottom*=sum([b.item(i)**2 for i in range(b.size)])**.5
    if(bottom==0):
        return 0
    
    return top/bottom

def hungarian(a,b):
    similarities = [[cosineSimilarity(a[i],b[j]) for j in range(len(b))] for i in range(len(a))]
    similarities = np.array(similarities)

    row,col = linear_sum_assignment(-1*similarities)   

    s = 0
    for i in range(len(row)):
        s+=similarities[row[i],col[i]]

    return s                

def addZeros(arr,num):
    if(num<=0):
        return arr
    #Add zero rows to end of array
    arr = np.append(arr,np.zeros((3,arr.shape[1])),axis=0)
    return arr

def dimensionReduction(M):
    total = np.sum(M)

    useRows = []
    for i in range(M.shape[0]):
        if(np.sum(M[i])>=.01*total):
            useRows.append(i)

    return M[useRows,:]
        
