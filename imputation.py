import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from entropy import runIterations
from math import e
from multiplicative_update import multiplicativeUpdate

def impute(V,k,epsilon=e**-12,iterations=10):
    tot = 0
    for q in range(iterations):
        Vtrain, Vtest = train_test_split(V.T, shuffle=False)
        Vtrain = Vtrain.T #119x72
        Vtest = Vtest.T #119x24 

        #119x5, 5x72
        Wtrain, Htrain = multiplicativeUpdate(Vtrain,k)

        Vtesthat = []

        d = Vtrain-Wtrain.dot(Htrain)

        #For each sample
        for category in range(Vtest.shape[0]):
            #Delete that row from V
            newV = np.delete(Vtest, category, axis=0)

            #Delete that row from W
            Wtest = np.delete(Wtrain, category, axis=0)

            #Find a new H
            Htest = np.linalg.lstsq(Wtest,newV,rcond=None)[0]
            Htest[Htest < 0] = 0

            #119x24
            newV = Wtrain.dot(Htest)
            col = newV[category]
            #print(newV[category],Vtest[category])
            Vtesthat.append(col)


        diff = Vtest-Vtesthat
        
        rec_error = mean_squared_error(Vtest, Vtesthat)
        tot+=rec_error

    return tot/iterations

