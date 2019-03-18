import numpy as np
import sys
sys.path.insert(0, '../data/alexsandrov/scripts/')

from convert_format import convert_format

signature_file = "signatures.npy"
data_file = "data.npy"

m = 100
n = 20

CANCER = "cervix"

def create_data(m=m,n=n,rank=6):
    W = np.random.random((m,rank))*4
    H = np.random.random((rank,n))

    data = W.dot(H)
    data+=np.random.normal(loc=0.1,scale=0.05,size=(m,n))
    np.save(signature_file,H)
    np.save(data_file,data)

def create_alexsandrov_data(cancer = CANCER):
    convert_format(cancer,header="../data/alexsandrov/scripts/")
