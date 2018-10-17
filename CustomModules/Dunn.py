# - author: ManuMarvel96

# calculating dunn index
# input should contain data points and cluster model
# returns a float vaue of dunn index

# works for sklearn KMeans
# works for kmedoids (here)

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def cal_minDistance(points):

    min_distance = float("inf")

    # 2d array of distances from one to another 
    distances = pairwise_distances(points)

    num_points = len(points)
    for i in range(num_points):
        for j in range(num_points):

            # a point to itself (distance) is zero 
            if distances[i][j] < min_distance and distances[i][j] != 0:
                min_distance = distances[i][j]

    return min_distance


def cal_maxDistance(data, indices):

    max_distance = 0

    # 2d array of distances from one to another
    distances = pairwise_distances(data[indices])

    num_points = len(indices)
    for i in range(num_points):
        for j in range(num_points):

            if distances[i][j] > max_distance:
                max_distance = distances[i][j]

    return max_distance      


def cal_dunnIndex(data, centroids, labels):

    min_interclusterDis = cal_minDistance(centroids)
    num_labels = len(set(labels))

    # grouping each cluster
    cluster_dataIndices = [[] for i in range(num_labels)]
    for i in range(len(labels)):
        cluster_dataIndices[labels[i]].append(i)

    # max distance of a pair of all clusters
    max_Pairdistance = 0
    for i in range(num_labels):
        dis = cal_maxDistance(data, cluster_dataIndices[i])
        if dis > max_Pairdistance:
            max_Pairdistance = dis
    
    dunn_index = min_interclusterDis / max_Pairdistance
    return dunn_index

    
