# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:20:13 2022

@author: will_
"""

import numpy as np

x = np.array(
    [[1,2,3,4],
    [5,5,6,7],
    [1,4,2,3],
    [5,3,2,1],
    [8,1,2,2]])

means = np.mean(x,axis=0)
stdevs = np.std(x,axis=0)
x_norm = (x - means) / stdevs
x_cov = np.cov(x_norm)
x_eig_vals,x_eig = np.linalg.eig(x_cov)
out_mat = x_norm.transpose() @ x_eig[:,0:2]