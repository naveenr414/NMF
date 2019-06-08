import numpy as np

def get_data_exome(cancer,header=""):
    cancer_file = open(header+"../exome/"+cancer+".txt").read().split("\n")[1:-1]
    cancer_file = [list(map(float,i.split("\t")[1:])) for i in cancer_file]

    assert len(cancer_file) == 96

    return np.array(cancer_file)
