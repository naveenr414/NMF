import numpy as np
import scipy

def permute_randomly(weight_matrix):
    """ Given an adjacency matrix, permute the edge weights"""
    values = []
    for i in range(len(weight_matrix)):
        for j in range(i):
            if weight_matrix[i][j] != 0:
                values.append(weight_matrix[i][j])

    np.random.shuffle(values)
    r = 0
    for i in range(len(weight_matrix)):
        for j in range(i):
            if weight_matrix[i][j] != 0:
                weight_matrix[i][j] = values[r]
                weight_matrix[j][i] = values[r]
                r+=1
    
    return weight_matrix

def normalize_rows(data):
    data = data.astype(np.float32)
    for i in range(len(data)):
        data[i]=np.divide(data[i],np.sum(data[i]))

    return data

def normalize(data):
    data = data.astype(np.float32)
    return data/np.sum(data)

def row_distances(data):
    """ Find the euclidean distance between every pair of rows """
    rows = data.shape[0]
    distances = np.zeros((rows,rows))
    for i in range(rows):
        for j in range(rows):
            distances[i][j] = np.linalg.norm(data[i]-data[j])

    return distances

def nearest_neighbor(data,k):
    """ Find the k nearest neighbor for each row """
    rows = data.shape[0]
    distances = row_distances(data)

    graph = np.zeros((rows,rows))
    for row in range(rows):
        smallest_indices = np.argpartition(distances[row],k)[:k+1]
        for i in smallest_indices:
            graph[row][i] = 1

    return graph

def epsilon_neighbor(data,epsilon):
    """ Find the nearest neighbor, so that each node is roughly connnected to epsilon percent of neighbors """
    rows = data.shape[0]
    distances = row_distances(data)
    all_distances = sorted(data.flatten())
    cutoff = all_distances[int(epsilon*len(all_distances))]

    graph = np.zeros((rows,rows))
    for i in range(rows):
        for j in range(rows):
            if(distances[i][j] < cutoff):
                graph[i][j] = 1
    
    return graph
