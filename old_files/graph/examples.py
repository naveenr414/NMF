import util
import numpy as np
import plotting
import matplotlib.pyplot as plt
from entropy import *
from math import log
from math import e
import copy

numTrials = 5
numIterations = 100

source = "alexsandrov"
V = 0
actualH = 0
k=5

def getAlexsandrovBreast():
    with open("data/alexsandrov/breast.txt") as f:
        data = f.read().split("\n")[1:-1]
        for i in range(0,len(data)):
            data[i] = data[i].split("\t")[1:]
            data[i] = list(map(int,data[i]))
    return np.array(data).T

def getAlexsandrovSignatures():
    signatureNums = [1,2,3,8,13]
    numList = []
    with open("data/alexsandrov/signaturesNew.txt") as f:
        data = f.read().split("\n")[1:]
        for i in range(len(data)):
            data[i] = list(map(float,data[i].split("\t")[3:]))
            temp = []
            for j in signatureNums:
                temp.append(data[i][j])
            numList.append(temp)
    return np.array(numList).T

if(source=="classdata"):
    V = util.readTSV("data/classdata/mutation-counts.tsv")
    actualH = np.load("data/classdata/example-signatures.npy")
elif(source=="alexsandrov"):
    V = getAlexsandrovBreast()
    actualH = getAlexsandrovSignatures()

def runTrials(mu,sigma,nuW,nuH,epsilon,distribution=0,method=1,randomDistro="Normal"):
    bestScore = 0
    for trial in range(numTrials):
        W,H = runIterations(V,k,numIterations=numIterations)


        score = plotting.cosineTable(actualH,H,showBest=True,draw=False)
        bestScore+=score


    bestScore/=numTrials

    return bestScore


def plotEntropyMethods(testEpsilons,nuH=1,nuW=1,mu=1,sigma=1):
    #Scores for each of the methods to get rid of negatives
    scoresZero = []

    for epsilon in testEpsilons:
        bestScore = runTrials(mu,sigma,nuW,nuH,epsilon)
        scoresZero.append(bestScore)

    #We plot a log scale for epsilons        
    testEpsilons = list(map(lambda x: log(x+10**(-40)),testEpsilons))

    testEpsilons = np.array(testEpsilons)
    scoresZero = np.array(scoresZero)

    plt.scatter(testEpsilons,scoresZero,c='b')
    plt.xlabel("Log epsilon")
    plt.ylabel("Cosine similarity score")

    return scoresZero

def findBestStarting(muList,sigmaList,method=0,distribution=1,nuW=1,nuH=1,epsilon=e**-5):    
    cosineMatrix = np.zeros((len(muList),len(sigmaList)))

    for a,mu in enumerate(muList):
        for b,sigma in enumerate(sigmaList):
            bestScore = runTrials(mu,sigma,nuW,nuH,epsilon)
            cosineMatrix[a,b] = bestScore

    return cosineMatrix

def findBestNu(nuWList,nuHList,epsilon=e**(-20),mu=1,sigma=1):
    cosineMatrix = np.zeros((len(nuWList),len(nuHList)))
    for a,nuW in enumerate(nuWList):
        for b,nuH in enumerate(nuHList):
            bestScore = runTrials(mu,sigma,nuW,nuH,epsilon)
            cosineMatrix[a,b] = bestScore
            
    return cosineMatrix
