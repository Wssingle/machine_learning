# -*- coding: utf-8 -*-
"""
Created on Sun May  8 00:01:50 2022

@author: will_
"""

"""
    Random forest regression tree
"""

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class Node(object):
    def __init__(self,features=None,targets=None,Id=0):
        self.Id = Id
        self.left_node = None
        self.right_node = None
        self.features = features
        self.targets = targets
        self.split = None
        self.split_var = None

class RFRT(object):
    def __init__(self,num_nodes=100):
        self.num_nodes = num_nodes
        self.head = None
    
    def train(self,X,y):     
        y = y[:,np.newaxis]
        #data = np.concatenate((X,y[:,np.newaxis]),axis=1)
        head = Node(features=X,targets=y)
        frontier = [head]
        
        # how many nodes
        for k in range(self.num_nodes):
            curr = frontier.pop(0)
            data = np.concatenate((curr.features,curr.targets),axis=1)
            best_split_loss = np.inf
            best_split = 0
            best_split_col = 0
            # For each feature vect
            for n in range(X.shape[1]):  
                col = n #np.random.randint(0,X.shape[1])
                
                # we sort the feature vector and targets together
                sort = data[data[:, col].argsort()]
                # For each split point
                for m in range(sort.shape[0]-1):
                    split_point = (sort[m][col] + sort[m+1][col]) / 2.0
                    loss = self._loss(split_point,sort[:,0:-1],(sort[:,-1])[:,np.newaxis],col)
                    if loss < best_split_loss:
                        best_split_loss = loss
                        best_split = split_point
                        best_split_col = col
                        #print(best_split)
                        
            ## We now add to tree the new split point
            l,r,lt,rt = self._find_split(best_split,curr.features,curr.targets,best_split_col)
            curr.split = best_split
            curr.split_var = best_split_col
            curr.left_node = Node(features=l,targets=lt,Id=k)
            curr.right_node = Node(features=r,targets=rt,Id=k)
            frontier.append(curr.left_node)
            frontier.append(curr.right_node)
        
        self.head = head
        return head
            
    def _find_split(self,split,samples,targets,col):
        l =  samples[samples[:,col]<split]
        r = samples[samples[:,col]>=split]
        lt = targets[samples[:,col]<split]
        rt = targets[samples[:,col]>=split]
        return l,r,lt,rt
        
    def _loss(self,split,samples,targets,col):
        left,right,lt,rt = self._find_split(split,samples,targets,col)
        
        l = np.mean(lt)
        r = np.mean(rt)
        
        
        if len(left) > 0:
            e1 = self._MSE(lt,l)
        else:
            e1 = 0
        
        if len(right) > 0:
            e2 = self._MSE(rt,r)
        else:
            e2 = 0
        
        return e1 + e2
        
    def _MSE(self,targets,guess):
        MSE = 1/len(targets)*np.sum((targets-guess)**2)
        return MSE

    def test(self,X_test,y_test):
        guess = []
        for indx,val in enumerate(X_test):
            curr = self.head
            while curr:
                if curr.split is None:
                    guess.append(np.mean(curr.targets))
                    break
                if val[curr.split_var] < curr.split:
                    if curr.left_node:
                        curr = curr.left_node
                    else:
                        print("None")
                elif val[curr.split_var] >= curr.split:
                    if curr.right_node:
                        curr = curr.right_node
                    else:
                        print("None")
                else:
                    print("ERROR")
        return guess

if __name__ == "__main__":
    num_features = 2
    num_trees = 100
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    
    # Make a regression problem
    X,y = make_regression(n_samples=100,n_features=num_features,noise=10)
    #ax.scatter(X[:,0],X[:,1],y)
    #plt.title("Regression Problem")
    #plt.show()

    # Get 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    guess_list = []
    
    # Bootstrap aggregation
    for i in range(num_trees):
        feature_list = []
        target_list = []
        for k in range(len(X_train)):
            val = np.random.randint(0,len(X_train))
            feature_list.append(X_train[val,:])
            target_list.append(y_train[val])
    
        tree = RFRT(num_nodes=15)
        head = tree.train(np.array(feature_list),np.array(target_list))
        #head = tree.train(X_train,y_train)
        
        guesses = tree.test(X_train,y_train)
        guess_list.append(guesses)

    final_g = np.sum(np.array(guess_list),axis=0)/len(guess_list)
    #final_g = stats.mode(guess_list).mode
    
    #tree = RT(num_nodes=1000)
    #head = tree.train(X_train,y_train)
    #guesses = tree.test(X_test,y_test)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X_test[:,0],y_test)
    #ax.scatter(X_test[:,0],guesses)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X_test[:,0],X_test[:,1],y_train)
    ax.scatter(X_train[:,0],X_train[:,1],y_train)
    ax.scatter(X_train[:,0],X_train[:,1],final_g)
    plt.title("My Regression Problem")
    plt.show()
    