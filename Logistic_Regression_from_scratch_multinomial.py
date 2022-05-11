# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:15:01 2022

@author: will_
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pdb

# TODO : I did not do backprop calc for bias terms, you will need to 
# do this math and add the code to do backprop

"""
If you have a multi-label classification problem = there is more than one 
"right answer" = the outputs are NOT mutually exclusive, then use a sigmoid 
function on each raw output independently. The sigmoid will allow you to have 
high probability for all of your classes, some of them, or none of them.

If you have a multi-class classification problem = there is only one
 "right answer" = the outputs are mutually exclusive, then use a softmax
 function. The softmax will enforce that the sum of the probabilities of your
 output classes are equal to one, so in order to increase the probability of 
 a particular class, your model must correspondingly decrease the probability 
 of at least one of the other classes.
"""

class DataLoader(object):
    def __init__(self,X,y):
        self._X = X
        self._y = y
    
    def get_random_sample(self):
        indx = np.random.randint(0,len(self._y))
        return self._X[indx], self._y[indx]
        
class MyLogisticRegression(object):
    def __init__(self,num_features,num_out,alpha=0.005,iters=1000,threshold=0.5):
        self._theta = np.random.random([num_out,num_features])
        self._bias = np.random.random([num_out,1]).flatten()
        self.alpha = alpha
        self.iters = iters
        self.epsilon = 10**-8
        self.threshold = threshold
        
    def _sigmoid(self,sample):
        return 1 / (1 + np.exp(-1*sample))
    
    def _softmax(self,sample):
        return np.exp(sample) / np.sum(np.exp(sample))
    
    def _dot(self,sample):
        return np.dot(self._theta,sample) #+ self._bias
            
    def _loss(self,x_hat,y_hat):
        if y_hat == 0:
            loss = 1*np.log(x_hat+self.epsilon)
        else:
            loss = -1*np.log(1-x_hat+self.epsilon)
        return loss
    
    def _soft_loss(self,x_hat,y):
        return -1 * np.log(x_hat[y])

    def _update(self,one_hot,out_soft,sample):
        # The math here is just wizardry done by backprop math
        sample = np.transpose(np.expand_dims(sample,axis=1)) 
        inner = np.expand_dims(one_hot.flatten()-out_soft,axis=1)
        update_theta_mat = -1*(inner @ sample)
                
        self._theta = self._theta - self.alpha*update_theta_mat
        #self._bias = self._bias - self.alpha*update_bias_mat
            
    def train(self,X_train,y_train):
        loader = DataLoader(X_train,y_train)
        for k in range(self.iters):
            sample,target = loader.get_random_sample()
            out_dot = self._dot(sample)
            out_soft = self._softmax(out_dot)
            loss = self._soft_loss(out_soft,target)
            
            one_hot = np.zeros([self._theta.shape[0],1])
            one_hot[target] = 1
            self._update(one_hot,out_soft,sample)
    
    def score_model(self,test_samples,test_targets,metric="acc"):
        # Can add f1, precision and recall here
        if len(test_samples) != len(test_targets):
            raise Exception("Input lengths should be the same")
        
        correct = 0
        for indx,sample in enumerate(test_samples):
            prediction = self._predict(sample)
            if prediction == test_targets[indx]:
                correct += 1
        return correct / len(test_targets)
    
    def _predict(self,sample):        
        out_dot = self._dot(sample)
        out_soft= self._softmax(out_dot)
        return np.argmax(out_soft)
   
if __name__ == "__main__":
    
    n_classes = 3
    num_features = 3
    
    # Make a classification problem with 2 features
    X,y = make_classification(n_samples=100,n_features=num_features,n_redundant=0,n_classes=n_classes,n_clusters_per_class=1,n_informative=num_features)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("Classification Problem")
    plt.show()

    # Get 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X, y)
    print(f"Scikit-learn score : {clf.score(X_test,y_test)}")
    
    my_log_reg = MyLogisticRegression(num_features,n_classes,alpha=0.005,iters=1000,threshold=0.5)
    my_log_reg.train(X_train,y_train)
    score = my_log_reg.score_model(X_test,y_test)
    print(f"My score : {score}")