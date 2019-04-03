import numpy as np
import sys

signature_file = "signatures.npy"
exposures_file = "exposures.npy"
data_file = "data.npy"

m = 100
n = 20

def create_data(m=m,n=n,rank=6):
    W = np.random.randint(0,10,(m,rank))
    H = np.random.random((rank,n))
    H[H<0.2] = 0
    H*=2
    
    data = W.dot(H)
    data+=np.random.normal(loc=0.1,scale=0.05,size=(m,n))
    np.save(signature_file,H)
    np.save(data_file,data)
    np.save(exposures_file,W)

