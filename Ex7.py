# -*- coding: utf-8 -*-
"""
@author: Nil

"""
import numpy as np
import time
import sys
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import GaussianNB

def function(model,data_3_and_8,target_3_and_8):
    
    """ 2-Fold Cross validation section: """
    # First, the data set is splitted into 2 equal sets:
    x_train,x_test,y_train,y_test=\
        train_test_split(data_3_and_8,target_3_and_8, test_size=0.5,random_state=0)
    
    # Here 2-fold Cross validation is effectuated:
    prediction = model.fit(x_train,y_train).predict(x_test)
    error_1 = (y_test != prediction).sum()
    print("1st Split. Number of errors : %d" % error_1)
    
    prediction = model.fit(x_test,y_test).predict(x_train)
    error_2 = (y_train != prediction).sum()
    print("2nd Split. Number of errors : %d" % error_2)
    
    error_3= (error_2+error_1)/7.14
    print("2-fold. Percentage prediction error (Real optimism) : %f" % error_3)
    
    """ Model selection section: """
    
    # 10-fold Cross-validation:
    kf = KFold(data_3_and_8.shape[0], n_folds=10)
    error=0
    for train, test in kf:
        a,Label_train,Label_test=0,[],[],
        Train,Test = np.zeros((321,64)), np.zeros((35,64))
        for x in range(321):
            Train[a]=data_3_and_8[train[x]]
            Label_train.append(target_3_and_8[train[x]])
            a+=1
        a=0
        for x in range(35):
            Test[a]= data_3_and_8[test[x]]
            Label_test.append(target_3_and_8[test[x]])
            a+=1
        prediction = model.fit(Train,Label_train).predict(Test)
        error = (Label_test != prediction).sum()+error
    error=error/35.7
    print("\n10-fold. Percentage prediction error : %f" % error)
    """
    # Reverse 10-fold Cross-validation:
    error=0
    for train, test in kf:
        a,Label_train,Label_test=0,[],[],
        Train,Test = np.zeros((321,64)), np.zeros((35,64))
        for x in range(321):
            Train[a]=data_3_and_8[train[x]]
            Label_train.append(target_3_and_8[train[x]])
            a+=1
        a=0
        for x in range(35):
            Test[a]= data_3_and_8[test[x]]
            Label_test.append(target_3_and_8[test[x]])
            a+=1
        prediction = model.fit(Test,Label_test).predict(Train)
        error = (Label_train != prediction).sum()+error
    error=error/35.7
    print("\n10-fold. Percentage prediction error : %f" % error)
    """
    
def Dimension_Reduction(Images,target):    
    
    Image=[]
    for x in range(1617):
        if target[x]==3 or target[x]==8:
            Image.append([Images[x][6][6],Images[x][6][4]])
    return Image
    
def main():  
    
    digits=load_digits(9)

    data = digits.data
    target = digits ["target"]
    images = digits.images
 
    # This function tell us that data contains 1797 images formed by 8*8 pixels. 
    print data.shape
    
    # In the following code, arrays that only contain the handwritten numbers 3 
    # and 8 are created for the set.   
    data_3_and_8 = np.zeros((357,64))
    labels_3_and_8=[]
    a=0
    for x in range(1617):
        if target[x]==3 or target[x]==8:
            data_3_and_8[a]=data[x]
            labels_3_and_8.append(target[x])
            a+=1
    
    # Each classifier goes to a common function where its error is tested by
    # 2-fold Cross validation and model selection criteria are applied.
    print "\n3-Nearest Neighbors classifier:"
    function(KNeighborsClassifier(n_neighbors=3, weights='distance'),data_3_and_8,labels_3_and_8)
    print "\nLinear Discriminant Analysis classifier:" 
    function(LDA(),data_3_and_8,labels_3_and_8)
    print "\nQuadratic Discriminant Analysis classifier:"
    function(QDA(),data_3_and_8,labels_3_and_8)
    print "\nNaive Bayes classifier:"
    function(GaussianNB(),data_3_and_8,labels_3_and_8)
    
    images_3_and_8=Dimension_Reduction(images,target)
    



main()