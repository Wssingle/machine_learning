# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:14:56 2022

@author: will_
"""

# The KNN should really use a Kd tree or Ball tree when the size of dataset
# gets large. For small data sets brute force distance metrics work well and
# are exact
# Also we can normalize data to make euclidean distance more fair accross other dims

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
#import pdb

class KNNClassifier(object):
    def __init__(self,train_set,train_set_targets,num_friends=3,dist_function=None,debug=False):
        self._X = train_set
        self._y = train_set_targets
        self._num_friends = num_friends
        
        # Can put in distance metric of choice manhattan etc...
        if dist_function is not None:
            self.dist_function = dist_function
        else:
            pass
        
        self.debug = debug
    
    def score_model(self,targets,predictions,metric="acc"):
        # Can add f1, precision and recall here
        if len(targets) != len(predictions):
            raise Exception("Input lengths should be the same")
        
        correct = 0
        for indx,target in enumerate(targets):
            if target == predictions[indx]:
                correct += 1
        return correct / len(targets)
    
    def batch_predict(self,sample_list):
        output_list = []
        for sample in sample_list:
           output_list.append(self._predict(sample))
        return output_list
        
    def _predict(self,sample):
        sorted_distance_indx = self._get_distances(sample)
        friends_near_sample = self._X[sorted_distance_indx]
        classes_of_friends = self._y[sorted_distance_indx]
            
        if self.debug:
            self._debug_plots(friends_near_sample,classes_of_friends)

        return self._classify(classes_of_friends)

    def _get_distances(self,sample):
        return np.argsort(np.sqrt(np.sum(np.power(self._X - sample,2),axis=1)))[0:self._num_friends]
    
    def _classify(self,classes_of_frineds):
        return np.round(np.sum(classes_of_frineds)/(len(classes_of_frineds)))
    
    def _debug_plots(self,friends_near_sample,classes_of_friends):
        plt.scatter(self._X[:,0],self._X[:,1],c=self._y)
        plt.scatter(friends_near_sample[:,0],friends_near_sample[:,1],color='g')
        plt.scatter(sample[0],sample[1],color='b')
        plt.show()

if __name__ == "__main__":
    # Make a classification problem with 2 features
    X,y = make_classification(n_samples=100,n_features=2,n_redundant=0)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()
    
    # Get 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_neighbors = 5
    sample = np.array([0,0])
    KNN = KNNClassifier(X_train,y_train,3)
    y_hat = KNN.batch_predict(X_test) 
    score = KNN.score_model(y_test,y_hat)
    print(f"my Score : {score}")
    
    # Train classifier
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(X_train, y_train)
        
    # Test Classifier
    print(f"Scikit-learn score : {neigh.score(X_test,y_test)}")