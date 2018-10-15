from util import createNearestNeighbor
import numpy as np
from pyvis.network import Network
import random
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import glob
from copy import deepcopy
from scipy.cluster.vq import kmeans2

SAMPLES = 100
CANCER_TYPES = 10 # How many of the types of cancer should we use
normalize = True
outliers = True

markerStyles = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

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

def removeOutliers(mat):
    mat = deepcopy(mat)

    for i in range(mat.shape[1]):
        mat = mat[abs(mat[:,i] - np.mean(mat[:,i])) < 3 * np.std(mat[:,i])]

    return mat

def printCancers():
    """Prints out the name of each cancer type"""
    
    for i in range(len(cancerNames)):
        print(i,cancerNames[i])

def toArray(fileName):
    """Converts exome data to an NP array """
    
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

def createNN():
    """ Creates a nearest neighbor graph
        Uses the augemented allData matrix"""
    
    nn = createNearestNeighbor(allData,k=5,metric="Cosine")
    g = Network()
    for cancer in range(len(cancerNames)):
        for nodeNumber in range(startingPositions[cancer],startingPositions[cancer+1]):
            g.add_node(nodeNumber,color=colors[cancer])

    for i in range(nn.shape[0]):
        for j in range(nn.shape[1]):
            if(nn[i][j]):
                g.add_edge(i,j)

    g.show("basic.html")

def createTSNE():
    """ Creates a TSNE graph """
    
    embedded = TSNE(n_components=2).fit_transform(allData)
    if(not outliers):
        embedded = removeOutliers(embedded)

    clusters = kmeans2(embedded,4,minit='points')
    labels = clusters[1]
    clusters = clusters[0]
    

    cancerType = 0
    for i in range(embedded.shape[0]):
        if(startingPositions[cancerType+1]==i):
            cancerType+=1

        plt.scatter(embedded[i][0],embedded[i][1],c=colors[cancerType],label=cancerNames[cancerType],marker=markerStyles[labels[i]])

    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)


    plt.scatter(clusters[:,0],clusters[:,1],s=80,c="red")
    plt.legend(newHandles,newLabels)
    plt.show()

startingPositions = []
allData = np.empty((0,0),dtype=np.float)

alexandrovFiles = glob.glob("alexandrov_data/*.txt")
if(CANCER_TYPES<len(alexandrovFiles)):
    alexandrovFiles = np.random.choice(alexandrovFiles,CANCER_TYPES,replace=False)
CANCER_TYPES = len(alexandrovFiles)

cancerNames = [i.split("\\")[-1].replace("_","").split("exome")[0] for i in alexandrovFiles]
printCancers()

colors = generateColors(len(cancerNames))

#Combines the data from each of the samples
for file in alexandrovFiles:
    cancerData = toArray(file)
    sampleRows = cancerData.shape[0]    
    selectedSamples = np.random.choice(sampleRows,min(sampleRows,SAMPLES),replace=False)

    if(len(startingPositions)==0):
        startingPositions.append(0)
    else:
        startingPositions.append(startingPositions[-1] + min(sampleRows,SAMPLES))
    
    cancerData = cancerData[selectedSamples]
    
    if(allData.size == 0): 
        allData = cancerData
    else:
        allData = np.append(allData,cancerData,axis=0)

startingPositions.append(allData.shape[0])

if(normalize):
    allData = normalizeRows(allData)

createTSNE()
