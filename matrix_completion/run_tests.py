from nmf_tensorflow import *
from copy import deepcopy
import numpy as np
import client_parser
import matplotlib.pyplot as plt
import util

def run_trials(data,weight,params,CROSS_VALIDATION=5):
    """ The cross validation is the number of groups that all pairs is split up into
    If cross validation = 5, then each time, 20% of the data is hidden"""
    
    rows,columns = data.shape
    number_elements = rows*columns
    all_pairs = [(i, j) for i in range(rows) for j in range(columns)]
    np.random.shuffle(all_pairs)

    results = []
    for i in range(CROSS_VALIDATION):
        new_params = deepcopy(params)
        start = int(i/CROSS_VALIDATION*number_elements)
        end = int((i+1)/CROSS_VALIDATION*number_elements)
        new_params['hidden_pairs'] = all_pairs[start:end]
        
        results.append(find_results(data,weight,new_params))

    return results

def grid_search(data,weight,lr_params,lamb_params,constant_params,iterations):
    """Grid searches through the data with weight_matrix
    Takes in the learning rate parameters (starting number
    and the distance or the range over which it searches)
    And the lambda parameters (ditto)
    And the constant parameters (rank)
    And the number of iterations"""

    
    possible_lr = lr_params['starting']
    num_lr = len(possible_lr)
    distance_lr = lr_params['distance']

    possible_lamb =  lamb_params['starting']
    num_lamb = len(possible_lamb)
    distance_lamb = lamb_params['distance']

    for iteration in range(iterations):
        score_matrix = np.zeros((len(possible_lr),len(possible_lamb)))
        for i,lr in enumerate(possible_lr):
            for j,lamb in enumerate(possible_lamb):
                new_params = deepcopy(constant_params)
                new_params['lr'] = 10**lr
                new_params['lambda'] = 10**lamb
                score_matrix[i][j] = run_trials(data,weight,new_params,CROSS_VALIDATION=2)
                score_matrix[i][j] = np.mean([i['imputation_error'] for i in results])

        min_i, min_j = np.where(score_matrix == np.min(score_matrix))
        min_lr = possible_lr[min_i[0]]
        min_lamb = possible_lamb[min_j[0]]

        distance_lr/=2
        distance_lamb/=2

        possible_lr = [k*distance_lr+min_lr
            for k in range(-num_lr//2,num_lr//2)]
        possible_lamb = [k*distance_lamb+min_lamb
            for k in range(-num_lamb//2,num_lamb//2)]

    return possible_lr[num_lr//2],possible_lamb[num_lamb//2]

def plot_errors(matrix_cost,graph_cost):
    """ Takes in an array of matrix_costs and graph_costs
    Plots matrix cost in red and graph cost in blue"""
    
    plt.scatter(range(len(matrix_cost)),matrix_cost,c='r',marker=',')
    plt.scatter(range(len(graph_cost)),graph_cost,c='b',marker=',')
    plt.show()

data,patients,categories = client_parser.parse_counts("data/brca_data.tsv")
weight,real_patients = client_parser.parse_dna("data/dna_brca.tsv",patients)
data = client_parser.select_rows(data,patients,real_patients)
weight = util.normalize(weight)
params = {'lr':10**-5,'lambda':10**4,'rank':5}

for lamb in range(0,6):
    params['lambda'] = 10**lamb
    trial = run_trials(data,weight,params,CROSS_VALIDATION=10)
    print(np.mean([i['imputation_error'] for i in trial]))
    plot_errors(trial[0]['matrix_error'],trial[0]['graph_error'])
