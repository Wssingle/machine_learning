# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:32:48 2019

@author: will_
"""

import tensorflow as tf
import numpy as np

######################################################## Kernel Regularizer

def custom_reg_upper_tri(weight_matrix):
   
    # Takes ---
    # weight_matrix: Tf variable of shape [height][width][channels][NumFilters]
    # lamda: the fudge factor for the loss contribution
    
    # Returns ---
    # Tensor that contributes to loss function of layer
    
    # Description ---
    # This function takes the wight matrix associated with a certain convolution layer...
    # and returns a loss tensor corresponding to the value of the abs cosine value of ...
    # all filters in conv layer  
   
    print('Upper Triangular Regularizer')
    
    
    shape = weight_matrix.shape #Gets shape of input
    #print('weight_matrix shape: %s' %weight_matrix.shape)  #Debug printf
    CxHxW = (shape[0]*shape[1]*shape[2]) #Flattened length is Channels*Height*Width
    Num_Filt = shape[3]                  #Number of filters is 3rd element
    #print('num Filt %d' %Num_Filt)       #Debug printf
    Z_tri = np.ones((Num_Filt, Num_Filt))
    Z_tri = np.triu(Z_tri,1)
    
    weight_matrix = tf.transpose(weight_matrix, [3,1,2,0])  #Rearranges to Numfilters first
    Filt_Rows = tf.reshape(weight_matrix, [Num_Filt,CxHxW]) #Flattens all weights
    Filt_Cols = tf.transpose(Filt_Rows)                     #Transposes flattened weights
    Filt_C_Norms = tf.norm(Filt_Cols , ord='euclidean', axis=0, keepdims=True) #Gets norm 
    Filt_R_Norms = tf.transpose(Filt_C_Norms) #Gets norm 


    Tprod = tf.math.reduce_sum( ( Z_tri * tf.math.abs( ( tf.math.divide( tf.tensordot( Filt_Rows , Filt_Cols, 1 ) , 
      tf.tensordot( Filt_R_Norms , Filt_C_Norms, 1)  )  ) ) ) )

    #Function: sum of all elems of abs( upper triangle * (dot product of V) / (outer product of V norms) )


    fptr = open(r'Lamda_File' , 'r') #Opens file to get lamda value
    lamda = float(fptr.read())
    fptr.close()
    
    lamda = lamda * 25.0
    print(lamda) #Debug printf
    
    return Tprod * lamda  #Output Tensor multiplied by fudge_factor

'''
    #To be determined ...
    
   
     
'''    

######################################################## End of Kernel Regularizer