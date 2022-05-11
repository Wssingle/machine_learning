# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:35:02 2022

@author: will_
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pdb

"""
This Linear regressor uses SGD, obviously you can just do MSE calculation
or you can use any other optimizer

To improve this algo you could vectorize all operations and do batch 
gradient descent

This algo uses L1 regularization. BTW derivitive of abs(x) is x/abs(x)
"""

class DataLoader(object):
    def __init__(self,X,y):
        self._X = X
        self._y = y
    
    def get_random_sample(self):
        indx = np.random.randint(0,len(self._y))
        return self._X[indx], self._y[indx]

class LinearReg(object):
    def __init__(self,iters=10000,alpha=0.005,lamb=0.1):
        self.lamb = lamb
        self.iters = iters
        self.alpha = alpha
        self.theta = None
        self.bias = None
        self.epsilon = 10**-7
        
    def _dot(self,x_vect):
        return (x_vect @ self.theta) + self.bias
        
    def _update(self,target_diff,x_vect):
        update_theta = -2*(target_diff)*x_vect + self.lamb*(self.theta / (np.abs(self.theta)+self.epsilon))
        update_bias = -2*(target_diff)
        
        self.theta = self.theta - self.alpha * update_theta 
        self.bias =  self.bias - self.alpha * update_bias
    
    def _loss(self,y_hat,y,show_loss=False):
        loss = (y -  y_hat)**2 + self.lamb * np.abs(self.theta) #MSE
        if show_loss:
            print(loss)
        return loss
    
    def train(self,X,y):
        loader = DataLoader(X,y)
        self.theta = np.random.random([1,X.shape[1]])
        self.bias = np.random.random([1,1])
        
        for k in range(self.iters):
            x_vect,target = loader.get_random_sample()
            y_hat = self._dot(x_vect)
            loss = self._loss(y_hat,target)
            self._update(target-y_hat,x_vect)
            
    def score(self,X_test,y_test):
        if len(X_test) != len(y_test):
            raise Exception("Input lengths should be the same")
        
        loss_count = 0
        for indx,x_vect in enumerate(X_test):
             y_hat = self._dot(x_vect)
             loss_count += self._loss(y_hat,y_test[indx])
        return loss_count / len(y_test)
    
if __name__ == "__main__":
    n_classes = 1
    num_features = 1
    
    # Make a regression problem
    X,y = make_regression(n_samples=100,n_features=num_features,noise=4.5)
    plt.scatter(X[:,0],y)
    plt.title("Regression Problem")
    plt.show()

    # Get 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = SGDRegressor()
    reg.fit(X_train,y_train)
    print(f"Scikit-learn score : score = {reg.score(X_test,y_test)}")
    
    my_reg = LinearReg()
    my_reg.train(X_train,y_train)
    score = my_reg.score(X_test, y_test)
    print(f"My score MSE : {score}")

    x = np.linspace(-4,4,100)
    out = []
    for val in x:
        out.append(my_reg._dot(np.array([val])))

    plt.scatter(X[:,0],y)
    plt.scatter(x,out)
    plt.title("My solution Regression Problem")
    plt.show()