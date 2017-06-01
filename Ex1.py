# -*- coding: utf-8 -*-
"""
Author: Nil Puig

"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier

def dist_loop(x_train,x_test):
    
    """ 
    "distance_matrix" is the Euclidian distance matrix between the training 
    and the test set. 
    """
    distance_matrix = np.empty((len(x_train),len(x_test)))
    
    for x in range(len(x_train)):
        for y in range(len(x_test)):
            x_all=zip(x_train[x],x_test[y])
            a=0
            for z in range(64):
                a+=np.square(x_all[z][0]-x_all[z][1])
            distance_matrix[x][y]=np.sqrt(a)
       
    return distance_matrix

def dist_vec(x_train,x_test):
    distance_matrix = np.empty((len(x_train),len(x_test)))
    # Even though this part was intented to do without a "for" loop, it has
    # been necessary to introduce at least the following one. However, the
    # performance hasn't been affected (The whole function only takes 0.21s)
    for x in range(len(x_train)):
        distance_matrix[x]=np.sqrt(np.sum(np.square(x_train[x]-x_test),axis=1))
    return distance_matrix
    
def nearest_neighbor_1(distance_matrix,x_train_1,x_test_1,y_train_1,y_test_1):
    d,y_pred=0,[]
    
    # the distance matrix is transposed:
    distance_matrix=distance_matrix.transpose()
    
    # The nearest point is selected and assigned:
    for x in range(len(x_test_1)):
        y_pred.append(y_train_1[np.argsort(distance_matrix[x])[0]])
        if y_test_1[x]==1:
            if y_pred[x]!=1:
                d+=1
    print ("    Number of mislabeled '1' : %d"%d)
    
def nearest_neighbor_k(distance_matrix,n,x_train_1,x_test_1,y_train_1,y_test_1,o,p):
    d,y_pred=0,[]
    # the distance matrix is transposed:
    distance_matrix=distance_matrix.transpose()
    
    # The nearest points are selected:
    for x in range(len(x_test_1)):
        a=0
        for y in range(n):
            if y_train_1[np.argsort(distance_matrix[x])[0]]==o:
                a+=1
            else:
                a=a-1
        if a>0:  
            y_pred.append(o)
        else:
            y_pred.append(p)
        if y_test_1[x]==o:
            if y_pred[x]!=o:
                d+=1
        else:
            if y_pred[x]!=p:
                d+=1
    plural=''
    if n>1: plural='s'
    print ("\n%d - Nearest neighbor%s classifier with numbers %d and %d:" 
        %(n,plural,o,p))
    print ("    Number of mislabeled points : %d, percentage error: %f "
        %(d,100*d/len(y_pred)))

def Cross_validation(n,data, target):
    split,count=len(data)/n,0

    # N splits are generated:
    for y in range (n):
        exec "split_%s = np.empty((split,64))" %(y)
        exec "y_split_%s = []" %(y)
        for x in range(split):
            exec "split_%s[x]=data[count]" %(y)
            exec "y_split_%s.append(target[count])" %(y)
            count+=1
 
    # For simplicity, I am going to use the K-NN classifier from sklearn:
    knn=KNeighborsClassifier(n_neighbors=3, weights='distance')
    mean,variance=0,0
    # The training sets are joined:
    for x in range(n):
        train,test = np.empty((split,64)),np.empty((split,64))
        y_train,y_test= [],[]
        num=0
        for y in range (n):
            if y!=x:
                if num==0:
                    exec "train=split_%s" %(y)
                    exec "y_train=y_split_%s" %(y)
                else:
                    exec "np.concatenate((train,split_%s), axis=0)" %(y)
                    exec "np.concatenate((y_train,y_split_%s), axis=0)" %(y)
                num+=1
            else: 
                exec "test=split_%s" %(y)
                exec "y_test=y_split_%s" %(y)
        exec "prediction = knn.fit(train,y_train).predict(test)"
        exec "error_%s = (y_test != prediction).sum()"%(x)
    # Finally, the mean and variance are found:
    for x in range(n):
        exec "mean+=error_%s"%(x)
    mean=mean/n
    for x in range(n):
        exec "variance+=np.square(mean-error_%s)"%(x)
    variance=variance/n
    print("%d - fold. Mean: %f, variance: %f" % (n, mean,variance))
    
def main():
    digits=load_digits()
    print digits.keys()

    data = digits["data"]
    images = digits ["images"]
    target = digits ["target"]
    target_names = digits ["target_names"]
     
    # This function tell us the data contains 1797 images formed by 8*8 pixels.    
    print data.shape

    
    #With these functions we can visualize one image of a "3"  
    f1=plt.figure('Image of a "3"')   
    plt.gray()
    plt.matshow(digits.images[3])
    f1.show()
    
    x_train,x_test,y_train,y_test=\
        cross_validation.train_test_split(data,target,
                                          test_size=0.4,random_state=0)
    """                      
    before=time.time()
    distance_matrix = dist_loop(x_train, x_test)
    after=time.time()
    
    print '\nDistance matrix:\n', distance_matrix
    print '\nRun time:'+repr(after-before)
    """
    
    # The distance matrix is computed using vectorization:    
    before=time.time()
    distance_matrix = dist_vec(x_train, x_test)
    after=time.time()
    
    print '\n\nDistance matrix:\n', distance_matrix
    print '\nRun time:'+repr(after-before),"\n"
    
    
    #With the following loop, the numbers 1 and 3 are filtered: 
    y_train_1_3,y_test_1_3=[],[]
    x_train_1_3=np.empty((222,64))
    x_test_1_3=np.empty((143,64))
    a,b=0,0
    for x in range(len(x_train)):
        if y_train[x]==1 or y_train[x]==3:
            x_train_1_3[a]=x_train[x]
            a+=1
            y_train_1_3.append(y_train[x])
        if x<len(x_test):
            if y_test[x]==1 or y_test[x]==3:
                x_test_1_3[b]=x_test[x]
                b+=1
                y_test_1_3.append(y_test[x])
    
    # the distance matrix between the numbers 1 and 3 is obtained:
    distance_matrix = dist_vec(x_train_1_3, x_test_1_3)
    print '\n\nDistance matrix:\n', distance_matrix
    # Here, a nearest neighbor classifier is implemented. It uses the 
    # previously distance matrix calculated.
    print ("\n1- Nearest neighbor classifier with numbers 1 and 3:")
    nearest_neighbor_1(distance_matrix,x_train_1_3, x_test_1_3,y_train_1_3,y_test_1_3)
    
    
    # We proceed in the same way with the numbers 1 and 7:
    y_train_1_7,y_test_1_7=[],[]
    x_train_1_7=np.empty((223,64))
    x_test_1_7=np.empty((138,64))
    a,b=0,0
    for x in range(len(x_train)):
        if y_train[x]==1 or y_train[x]==7:
            x_train_1_7[a]=x_train[x]
            a+=1
            y_train_1_7.append(y_train[x])
        if x<len(x_test):
            if y_test[x]==1 or y_test[x]==7:
                x_test_1_7[b]=x_test[x]
                b+=1
                y_test_1_7.append(y_test[x])
    

    distance_matrix = dist_vec(x_train_1_7, x_test_1_7)
    
    # Here, a nearest neighbor classifier is implemented. It uses the 
    # previously distance matrix calculated.
    print ("\n1- Nearest neighbor classifier with numbers 1 and 7:")
    nearest_neighbor_1(distance_matrix,x_train_1_7, x_test_1_7,y_train_1_7,y_test_1_7)
    
    # A "for" loop performs the nearest classifier with k=1,3,5,7 and 9:
    print ("\n\n          --------- Numbers 1 and 7 ----------\n")
    n,o,p=1,1,7
    for x in range(5):
        nearest_neighbor_k(distance_matrix,n,x_train_1_7, x_test_1_7,y_train_1_7,y_test_1_7,o,p)
        n+=2
    nearest_neighbor_k(distance_matrix,17,x_train_1_7, x_test_1_7,y_train_1_7,y_test_1_7,o,p)
    nearest_neighbor_k(distance_matrix,33,x_train_1_7, x_test_1_7,y_train_1_7,y_test_1_7,o,p)

    # Now, the same goes with numbers 1 and 7
    print ("\n\n          --------- Numbers 1 and 3 ----------\n")
    distance_matrix = dist_vec(x_train_1_3, x_test_1_3)
    n,o,p=1,1,3
    for x in range(5):
        nearest_neighbor_k(distance_matrix,n,x_train_1_3, x_test_1_3,y_train_1_3,y_test_1_3,o,p)
        n+=2
    nearest_neighbor_k(distance_matrix,17,x_train_1_3, x_test_1_3,y_train_1_3,y_test_1_3,o,p)
    nearest_neighbor_k(distance_matrix,33,x_train_1_3, x_test_1_3,y_train_1_3,y_test_1_3,o,p)
    print("\n")
    # The next function performs N- fold cross validation 
    n=2
    Cross_validation(n, data, target)
    n=5
    Cross_validation(n, data, target)
    n=10
    Cross_validation(n, data, target)

main()
