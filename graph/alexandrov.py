from util import createNearestNeighbor
import numpy as np
from pyvis.network import Network
import random
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def generateColors(n):
    colorList = []
    for i in range(n):
        tempColor = "#"
        for j in range(6):
            tempColor+=str(random.sample(list(range(10))+["A","B","C","D","E","F"],1)[0])
        colorList.append(tempColor)
    return colorList

cancerList = ["bladder","breast","cervix","chromophobe","clear_cell","CLL","colorectum","esophageal","glioblastoma","neck","ovary","prostate","stomach"]
for i in range(len(cancerList)):
    print(100*i,cancerList[i])

colors = generateColors(len(cancerList))
rows = 100

allData = np.empty((0,0),dtype=np.float)

for i in cancerList:
    with open("alexandrov_data/"+i+"_exome.txt") as f:
        cancerData = f.read().split("\n")[1:-1]
        for i in range(0,len(cancerData)):
            cancerData[i] = cancerData[i].split("\t")[1:]
            cancerData[i] = list(map(int,cancerData[i]))

    cancerData = np.array(cancerData,dtype=np.float).T

    cancerData = cancerData[np.random.choice(cancerData.shape[0], min(rows,cancerData.shape[0]), replace=False)]

    if(not allData.shape[0]):
        allData = cancerData
    else:
        allData = np.append(allData,cancerData,axis=0)

normalize = False
if(normalize):
    for i in range(allData.shape[0]):
        s = sum(allData[i])
        for j in range(allData.shape[1]):
            num = float(allData[i][j])/float(s)
            allData[i][j] = float(allData[i][j])
            allData[i][j]/=float(s)

            allData[i][j] = num

    

def createNN():
    nn = createNearestNeighbor(allData,k=5,metric="Cosine")
    g = Network()
    for cancer in range(len(cancerList)):
        for sample in range(rows):
            nodeNumber = cancer*rows+sample
            g.add_node(nodeNumber,color=colors[cancer])

    for i in range(nn.shape[0]):
        for j in range(nn.shape[1]):
            if(nn[i][j]):
                g.add_edge(i,j)

    g.show("basic.html")

def createTSNE():
    embedded = TSNE(n_components=2).fit_transform(allData)
    print(embedded.shape)
    for i, c, label in zip(range(len(cancerList)), colors, cancerList):
        print(i)
        plt.scatter(embedded[i*rows:(i+1)*rows, 0], embedded[i*rows:(i+1)*rows, 1], c=c, label=label)

    plt.legend()
    plt.show()

#createNN()
createTSNE()
