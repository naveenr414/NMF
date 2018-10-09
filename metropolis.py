import numpy as np
import random
from scipy.stats import norm

def randomSample(f,trials=1000,x=0):
    """Uses the Metropolis-Hastings algorithm to sample from a probabiltiy distro"""
    sampleList = [x]
    for i in range(trials):
        x = sampleList[-1]
        xPrime = np.random.normal(x,1)

        alpha = 1
        if(f(x)>f(xPrime)): 
            alpha = f(xPrime)/f(x)

        u = random.random()
        if(u>alpha):
            sampleList.append(x)
        else:
            sampleList.append(xPrime)

    return sampleList[500:][::100]

    
