# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:33:29 2022

@author: will_
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pdb

# TODO K++ is not implemented

"""
https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/

- Normalization is good for clustering as the data is geometric
- We determine the number of clusters to use by creating an elbow diagram
    using the intra-cluster min distance as metric.
- We can also use Dunn index which is Min(inter clust dist) / Max(intra clust dist)
- Stopping criterion can be num iterations, clusters dont move anymore, or 
    no new points are assigned to new clusters
- Additions to KMeans are Kmeans++ and hierarchial clustering

"""

class MyKMeans(object):
    def __init__(self,num_centroids,max_iters=300,init_type="random"):
        self._c = num_centroids
        self._it = max_iters
        self._init = init_type
        self.centroids = None
        self.epsilon = 10**-7
    def _K_plus_init(self,X):
        pass
    
    def _random_init(self,X):
        mins = np.min(X,axis=0)
        maxs = np.max(X,axis=0)        
        return np.random.uniform(mins,maxs,(self._c,X.shape[1]))

    def _find_centroid_distances(self,centroids,X):
        p = np.zeros([X.shape[0],self._c])
               
        for indx,c in enumerate(centroids):
            p[:,indx] = np.linalg.norm(c-X,axis=1)
        return p
    
    def _find_points_for_cluster(self,centroids,distances,X):
        # We find which point is closest to each cluster
        # Then we add that point to counter
        
        p = np.zeros([self._c,X.shape[1]+1])
        sorted_vals = np.argsort(distances,axis=1)
        
        for indx,val in enumerate(sorted_vals):
            p[val[0],:-1] += X[indx] 
            p[val[0],-1] += 1
        
        max_manhattan_dist = np.max(np.sum(np.abs(p[:,0:-1]),axis=1))
        new_cluster_location = p[:,:-1] / (np.broadcast_to(np.expand_dims(p[:,-1],axis=1),(self._c, X.shape[1]))+self.epsilon)
        return max_manhattan_dist,new_cluster_location
        
    def fit(self,X):
        c = self._random_init(X)
        #c = self._K_plus_init(X)

        for k in range(self._it):
            distances = self._find_centroid_distances(c,X)
            max_manhattan_dist,c = self._find_points_for_cluster(c, distances,X)            

        self.centroids = c        
        return max_manhattan_dist,c

if __name__ == "__main__":
    n_classes = 2
    num_features = 2
    num_clusts = 20
    
    # Make a classification problem with 2 features
    X,y = make_classification(n_samples=100,n_features=num_features,n_redundant=0,n_classes=n_classes,n_clusters_per_class=1,n_informative=num_features)
    
    X = (X - X.mean(axis=0)) / (X.std(axis=0))
    
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("Classification Problem, Scikit-Learn Clustering")

    max_iter = 500
    init = 'random' #'k-means++'
    kmeans = KMeans(max_iter=max_iter,n_clusters=num_clusts, random_state=0,init=init).fit(X)
    x = kmeans.cluster_centers_
    
    plt.scatter(x[:,0],x[:,1],color='r')
    plt.show()
    
    # Get 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    elbow = []
    for k in range(2,num_clusts+1):
        # My cluster
        algo = MyKMeans(num_centroids=k)
        max_manhattan_dist,centroids = algo.fit(X)
        elbow.append(max_manhattan_dist)
        
        plt.scatter(X[:,0],X[:,1],c=y)
        plt.title("Classification Problem, My Clustering")
        plt.scatter(centroids[:,0],centroids[:,1],color='r')
        plt.show()

    plt.plot(elbow)
    plt.title("Elbow graph of manhattan intra cluster error")
    plt.show()