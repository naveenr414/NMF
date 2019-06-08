from nmf_tensorflow import *
from copy import deepcopy
import numpy as np
import client_parser
import matplotlib.pyplot as plt
from multiprocessing import Pool
import util

def log_result(result):
    print(result['imputation_error'])

def run_trials(data,weight,params,CROSS_VALIDATION=5):
    """ The cross validation is the number of groups that all pairs is split up into
    If cross validation = 5, then each time, 20% of the data is hidden"""
    
    rows,columns = data.shape
    number_elements = rows*columns
    all_pairs = [(i, j) for i in range(rows) for j in range(columns)]
    np.random.shuffle(all_pairs)

    results = []
    start_list = [int(i/CROSS_VALIDATION*number_elements) for i in range(CROSS_VALIDATION)]
    end_list = [int((i+1)/CROSS_VALIDATION*number_elements) for i in range(CROSS_VALIDATION)]
    new_params = [deepcopy(params) for i in range(CROSS_VALIDATION)]
    for i in range(CROSS_VALIDATION):
        new_params[i]['hidden_pairs'] = all_pairs[start_list[i]:end_list[i]]        

    pool = Pool(processes=4)
    for i in range(CROSS_VALIDATION):
        pool.apply_async(find_results,args=(data,weight,new_params[i]))
    pool.close()
    pool.join()

    return True

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
weight,weight_patients = client_parser.parse_pathways("data/breast_pathways.tsv",["BRCA1","BRCA2","RAD50"])
real_patients = list(set(patients).intersection(set(weight_patients)))
data = client_parser.select_rows(data,patients,real_patients)
weight = client_parser.select_rows_and_columns(weight,weight_patients,real_patients)
weight = util.normalize(weight)
params = {'lr':10**-5,'lambda':0,'rank':5}

results = run_trials(data,weight,params,CROSS_VALIDATION=10)
