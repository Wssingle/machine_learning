# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:30:00 2022

@author: will_
"""

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from scipy import stats

class Node(object):
    def __init__(self,features=None,targets=None,Id=0):
        self.Id = Id
        self.left_node = None
        self.right_node = None
        self.features = features
        self.targets = targets
        self.split = None
        self.split_var = None

class RFCT(object):
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
                # we sort the feature vector and targets together
                sort = data[data[:, n].argsort()]
                # For each split point
                for m in range(sort.shape[0]-1):
                    split_point = (sort[m][n] + sort[m+1][n]) / 2.0
                    loss = self._loss(split_point,sort[:,0:-1],(sort[:,-1])[:,np.newaxis],n)
                    if loss < best_split_loss:
                        best_split_loss = loss
                        best_split = split_point
                        best_split_col = n
                        #print(best_split)
                        
            ## We now add to tree the new split point
            l,r,lt,rt = self._find_split(best_split,curr.features,curr.targets,best_split_col)
            curr.split = best_split
            curr.split_var = best_split_col
            #print(len(lt))
            #print(len(rt))
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
        
        l_count = Counter(lt.flatten())
        r_count = Counter(rt.flatten())
        
        l_sum = sum(l_count.values())
        #print(f"l_sum: {l_sum}")
        r_sum = sum(r_count.values())
        #print(f"r_sum: {r_sum}")
        tot = l_sum + r_sum
        
        G1 = 1-(sum( (np.array(list(l_count.values())) / l_sum)**2))
        G2 = 1-(sum( (np.array(list(r_count.values())) / r_sum)**2))
        
        gini = l_sum/tot*G1 + r_sum/tot*G2              
       # print(gini)
        return gini

    def test(self,X_test,y_test):
        guess = []
        for indx,val in enumerate(X_test):
            curr = self.head
            while curr:
                if curr.split is None:
                    if len(stats.mode(curr.targets).mode) > 0:
                        stats.mode(curr.targets)
                        mode = stats.mode(curr.targets).mode[0][0]
                        guess.append(mode)
                    else:
                        guess.append(1000000000)
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
    num_trees = 200
    
    # Make a classification problem with 2 features
    X,y = make_classification(n_samples=1000,n_features=2,n_redundant=0,n_classes=2,n_informative=2)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("Classification Problem")
    plt.show()
    
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
    
        tree = RFCT(num_nodes=30)
        head = tree.train(np.array(feature_list),np.array(target_list))
        #head = tree.train(X_train,y_train)
        
        guesses = tree.test(X_test,y_test)
        guess_list.append(guesses)


    #final_g = np.sum(np.array(guess_list),axis=0)/len(guess_list)
    final_g = stats.mode(guess_list).mode
    
    
    plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
    plt.title("Classification Problem")
    plt.show()
    
    plt.scatter(X_test[:,0],X_test[:,1],c=final_g)
    plt.title("My Classification Problem")
    plt.show()