from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold
from statistics import mode
import numpy as np
from copy import deepcopy
from trifactor import trifactor

## Parse the file ## 
inpFile = "ppiData/BIOGRID-ORGANISM-Caenorhabditis_elegans-2.0.56.tab.txt"
f = open(inpFile)
for i in range(35):
    f.readline()

interactions = []

line = f.readline()
i = 0

while(line!=''):
    if("Phenotypic" in line):
        i+=1
    
    line = line.split("\t")
    interactorA = line[0]
    interactorB = line[1]

    interactions.append(interactorA + " "+interactorB)
    interactions.append(interactorB + " "+interactorA)

    line = f.readline()

## Turn the file into an adjacency Matrix ## 
nameToNumber = {}
currentNum = 0

for i in interactions:
    interactorA,interactorB = i.split(" ")
    if(interactorA not in nameToNumber):
        nameToNumber[interactorA] = currentNum
        currentNum+=1

    if(interactorB not in nameToNumber):
        nameToNumber[interactorB] = currentNum
        currentNum+=1

adj = [[0 for i in range(currentNum)] for j in range(currentNum)]

for i in interactions:
    interactorA,interactorB = i.split(" ")
    if(nameToNumber[interactorA]>nameToNumber[interactorB]):
        adj[nameToNumber[interactorB]][nameToNumber[interactorA]] = 1
    else:
        adj[nameToNumber[interactorA]][nameToNumber[interactorB]] = 1

## Compute the largest connected component ## 
csr = csr_matrix(adj)
_components, labels = connected_components(csgraph=csr, directed=False, return_labels=True)

## Take only the largest connected component ## 
bestLabel = mode(labels)
adj = np.array(adj)
selected = [i for i,label in enumerate(labels) if label == bestLabel]
selected = np.array(selected)
adj = adj[selected[:, None], selected]

for i in range(adj.shape[0]):
    adj[i,i] = 0

## Find where the adjacency has a 1 ##
locations = []
for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        if(adj[i,j] == 1):
            locations.append((i,j))

np.random.shuffle(locations)

## Perform K fold cross validation ##
kf = KFold(n_splits=20)
n = 0
for train, test in kf.split(locations):
    test = [locations[i] for i in test]
    train = [locations[i] for i in train]
    
    n+=1
    tempAdj = deepcopy(adj)
    for i in test:
        tempAdj[i] = 0

    factored = trifactor(tempAdj,train,200,iterations=200)

    for i in train:
        factored[tuple(i)] = 0

    flattened = factored.flatten()
    flattened.sort()

    print("HERE")
    
    for pick in range(4,500,4):

        cutoff = flattened[-(pick)]
        truePositive, falsePositive, falseNegative = 0,0,0
    
        indices = np.argwhere(factored>cutoff)

        for i in indices:
            if adj[tuple(i)] == 1:
                truePositive+=1
            else:
                falsePositive+=1

        for i in test:
            if tuple(i) not in indices:
                falseNegative+=1

        precision = 0
        if truePositive+falsePositive!=0:
            precision = truePositive/(truePositive+falsePositive)

        recall = 0
        if truePositive+falseNegative!=0:
            recall = truePositive/(truePositive+falseNegative)
        print(pick,recall,precision)
