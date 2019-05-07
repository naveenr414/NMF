from nmf_tensorflow import find_results
from create_data import create_data, m, n
import numpy as np
import matplotlib.pyplot as plt
import time
np.warnings.filterwarnings('ignore')

create_data()

def get_average_signature_error(rank):
    """ Input: Integer rank 
    Returns the average signature distance """
    results = [find_results(rank) for i in range(10)]
    results = [i["signature"] for i in results if i!= None]

    if(len(results) == 0):
        return 0
    
    return sum(results)/len(results)
    
def plot_reconstruction(rank):
    """ Input: List of Integers
        Output: Plot of rank vs norm error"""
    percentages = np.arange(0.1,0.9,0.05)
    norms = []
    
    for HIDDEN_PERCENT in percentages:
        norms.append(find_results(rank,HIDDEN_PERCENT=HIDDEN_PERCENT)["norm"])

    plt.plot(percentages,norms)
    plt.show()

def plot_reconstruction_normalized(rank):
    """ Input: List of Integers
    Output: Plot of rank vs norm error (normalized) """

    percentages = np.arange(0.1,0.9,0.05)
    norms = []
    
    for HIDDEN_PERCENT in percentages:
        norms.append(find_results(rank,HIDDEN_PERCENT=HIDDEN_PERCENT)["norm"]/(m*n*HIDDEN_PERCENT))

    plt.plot(percentages,norms)
    plt.show()

def plot_signature_norm(rank_range):
    """ Input: List of Integers
        Output: Plot of signature error vs norm error"""

    signatures = []
    norms = []
    for rank in rank_range:
        results = find_results(rank)
        signatures.append(results["signature"])
        norms.append(results["norm"])

    plt.plot(norms,signatures)
    plt.show()

def percent_top(rank_range,trials=10,mask=True):
    """ Input: List of Integers
        Output: Percent of top 5 of norm intersecting
        With top 5 signatures"""


    results = []
    i = 0
    
    for i in range(trials):
        create_data()
        temp_data = []

        for rank in rank_range:
            if(mask):
                data = find_results(rank,HIDDEN_PERCENT=0.3)
                while data == None:
                    print("Restarting",rank)
                    data = find_results(rank,HIDDEN_PERCENT=0.3)
            else:   
                data = find_results_no_mask(rank)
            tuple_data = (data["norm"],data["signature"])
            temp_data.append(tuple_data)

        norm_sorted = sorted(temp_data,key = lambda x: x[0])
        signature_sorted = sorted(temp_data,key=lambda x: x[1])

        num_in_5 = 0
        for j in range(5):
            if(signature_sorted.index(norm_sorted[j])<5):
                num_in_5+=1
        results.append(num_in_5/5)
        print(results)

    return results

def pairs_correct(rank_range,HIDDEN_PERCENT=0.3,mask=True):
    """ Input: List of Integers
    Otuput: Percent of (rank,rank) pairs that correctly predicted
    Signature trend based off norm """
    
    temp_data = []
    
    for rank in rank_range:
        r = 0
        if mask:
            data = find_results(rank,HIDDEN_PERCENT=HIDDEN_PERCENT)
            while data == None:
                r+=1
                data = find_results(rank,HIDDEN_PERCENT=HIDDEN_PERCENT)
                if(r>40):
                    continue    
        else:
            data = find_results_no_mask(rank)
        tuple_data = (data["norm"],data["signature"],data["exposure"])
        temp_data.append(tuple_data)
     
    num_correct = 0
    num_total = 0

    for i in range(len(temp_data)):
        for j in range(i+1,len(temp_data)):
            num_total+=1

            if(temp_data[i][0]<=temp_data[j][0] and (temp_data[i][1]<=temp_data[j][1] or temp_data[i][2]<=temp_data[j][2])  or
               temp_data[i][0]>=temp_data[j][0] and (temp_data[i][1]>temp_data[j][1] or temp_data[i][2]>temp_data[j][2])):
                num_correct+=1
            else:
                print(temp_data[i])
                print(temp_data[j])
                print()

    return num_correct/num_total

def rank_correct(rank_range,HIDDEN_PERCENT=0.3,mask=True):
    """ Input: List of Integers
    Otuput: 0/1 for whether the correct rank is found """
    
    temp_data = []
    real_rank = 6
    
    for rank in rank_range:
        r = 0
        if mask:
            data = find_results(rank,HIDDEN_PERCENT=HIDDEN_PERCENT)
            while data == None:
                r+=1
                data = find_results(rank,HIDDEN_PERCENT=HIDDEN_PERCENT)
                if(r>40):
                    continue
        else:
            data = find_results_no_mask(rank)
        tuple_data = (rank,data["norm"],data["signature"])
        temp_data.append(tuple_data)

    if sorted(temp_data,key=lambda x: x[1])[0][0] == real_rank:
        return 1
    print(sorted(temp_data,key=lambda x: x[1])[0][0])
    return 0
    
def plot_pairs_correct(rank_range,percent_range):
    """ Input: List of Integers, List of Floats
        Output: Goes through every percent in percent_range
    Runs pairs_correct, and plots that"""
    
    pairs = [pairs_correct(rank_range,percent) for percent in percent_range]
    plt.plot(percent_range,pairs)
    plt.show()

print(pairs_correct([2,3,4,5,6,7,8,9,10,11]))
