from get_signatures import get_signatures
from get_data import get_data_exome 

import numpy as np

signature_file = "signatures.npy"
data_file = "data.npy"

def convert_format(cancer,header=""):
    signatures = np.array(get_signatures(cancer,header=header))
    np.save(signature_file,signatures)

    data = get_data_exome(cancer,header=header).T
    np.save(data_file,data)
