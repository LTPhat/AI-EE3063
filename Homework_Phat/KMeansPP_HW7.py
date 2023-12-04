# importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import datasets



def create_data(n_cluster, n_points_each_cluster, n_feature = 2):
    X, y, centers = datasets.make_blobs(n_samples=n_cluster * n_points_each_cluster, n_features=2, centers= n_cluster, return_centers=True)
    return X, y, centers

        
        
# function to plot the selected centroids
def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker = '.',
                color = 'gray', label = 'data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color = 'black', label = 'previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color = 'red', label = 'next centroid')
    plt.title('Select % d th centroid'%(centroids.shape[0]))
     
    plt.legend()
    plt.show()
          
# function to compute euclidean distance
def distance(p1, p2):
    return np.sum((p1 - p2)**2)
  
# initialization algorithm
def initialize(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
    plot(data, np.array(centroids))
  
    ## compute remaining k - 1 centroids
    for _ in range(k - 1):
         
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
             
            ## Calculate dist of all points to the nearest centroids
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        plot(data, np.array(centroids))
    return centroids
  

def plot_before(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker = '.',
                color = 'gray', label = 'data points')
    for centroi in centroids:
        plt.scatter(centroi[0], centroi[1],
                color = 'red', label = 'centroid')
    plt.title("Centroids from create dataset")
    plt.show()
    return 


if __name__ == "__main__":
    X, y, centers = create_data(4, 100)
    print("Initialized centroids:", centers)
    plot_before(X, centers)
    print(X.shape)
    centroids = initialize(X, k = 4)
