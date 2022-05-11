# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:35:28 2022

@author: will_
"""

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd

"""
In naive bayes we can have categorical or continuous values
For categorical values we just look at statistics Probability of class X
given variable takes on a,b,c etc.
For continuous variables we assume variables come from certain distro and 
then knowing mean and stdev we calculate likliehood from PDF. This is weird 
to do this beacuse the probsbility of any event is zero in PDF, but we use the 
PDF value as a representation using Naive bayes. So we just plug in the value
we get for the variable and then use the PDF likliehood as the "probability".
So we make a distribution for the continuous variable for all classes.
e.g is cat has a stdev and mean, and is dog has an stdev and mean.
"""

class NaiveBayes(object):
    def __init__(self):
        self.avgs = None
        self.stds = None
        self.class_probs = None
    
    def _guass(self,mean,std,x):
        return np.product(np.array(1/(std*np.sqrt(2*np.pi)) * np.exp((-1/2)*((x-mean)/std)**2)))
    
    def _get_stats(self,X,y):
        stack = np.hstack((X_train,y_train[:,np.newaxis]))
        df = pd.DataFrame(stack)
        class_probs = df[stack.shape[1]-1].value_counts() / df[stack.shape[1]-1].count()
        avgs = df.groupby([stack.shape[1]-1]).mean()
        stds = df.groupby([stack.shape[1]-1]).std()
        return avgs,stds,class_probs 
    
    def score_model(self,predictions,test_targets,metric="acc"):
        # Can add f1, precision and recall here
        if len(predictions) != len(test_targets):
            raise Exception("Input lengths should be the same")
        
        correct = 0
        for indx,sample in enumerate(predictions):
            if sample == test_targets[indx]:
                correct += 1
        return correct / len(test_targets)
    
    def predict(self,x_vect):
        out_list = []
        for vect in x_vect:
            prob_list = []
            for indx,prob in enumerate(self.class_probs):
                probs_sum = self._guass(self.avgs.loc[indx], self.stds.loc[indx], vect)
                prob_list.append(prob*probs_sum)
            out_list.append(np.argmax(prob_list))
            
        return out_list
        
    def fit(self,X,y):
        avgs,stds,class_probs = self._get_stats(X,y)
        self.avgs = avgs
        self.stds = stds
        self.class_probs = class_probs
    
if __name__ == '__main__':
    n_classes = 3
    num_features = 2
    
    # Make a classification problem with 2 features
    X,y = make_classification(n_samples=100,n_features=num_features,n_redundant=0,n_classes=n_classes,n_clusters_per_class=1,n_informative=num_features)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("Classification Problem")
    plt.show()

    # Get 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print(f"Scikit-learn score : score = {gnb.score(X_test,y_test)}")
    
    myBay = NaiveBayes()
    myBay.fit(X_train,y_train)
    labels = myBay.predict(X_test)
    score = myBay.score_model(labels,y_test)
    print(f"My score : {score}")
    
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.scatter(X_test[:,0],X_test[:,1],c=labels,cmap='Reds')
    plt.title("My Classifications")
    plt.show()