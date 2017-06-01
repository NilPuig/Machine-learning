# -*- coding: utf-8 -*-
"""
@author: Nil

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import cross_validation

def Dimension_Reduction(data):    
    data_dr=np.empty((len(data),2))   
    inter=[]
    for x in range(len(data)):
        inter=data[x] 
        data_dr[x]=np.array([inter[5],inter[60]])
    return data_dr

def Nearest_Mean(x1,y1,x2,y2,x_test):
    # Firs of all, the means are computed:
    mean_x1,mean_y1,mean_x2,mean_y2=np.sum(x1)/len(x1),np.sum(y1)/len(x1),np.sum(x2)/len(x2),np.sum(y2)/len(x2)
    
    #Standard deviations are found:
    stdx1 = np.std(x1)
    stdy1 = np.std(y1)
    stdx2 = np.std(x2)
    stdy2 = np.std(y2)
    std_1=[stdx1,stdy1]
    std_7=[stdx2,stdy2]
    #Now, we are going to predict the labels of the test set:
    a,b,prediction=0,0,[]
    for x in range(len(x_test)):
        a=np.square(x_test[x][0]-mean_x1)+np.square(x_test[x][1]-mean_y1)
        b=np.square(x_test[x][0]-mean_x2)+np.square(x_test[x][1]-mean_y2)
        if a>b:
            prediction.append(7)
        else:
            prediction.append(1)
    return prediction,std_1,std_7
    
def compute_qda(x_train, y_train_binary):
    n0,n1,p0, p1,s=0,0,0,0,x_train.shape[1]
    mu0, mu1,covmat0, covmat1= np.zeros((1)),np.zeros((1)),np.zeros((s,s)),np.zeros((s,s))      
    for x in range(len(y_train_binary)):
        if y_train_binary[x]==0:
            mu0=np.sum([mu0,x_train[x]], axis=0)
            n0+=1
        else:
            mu1=np.sum([mu1,x_train[x]], axis=0) 
            n1+=1
    mu0=mu0/(n0)
    mu1=mu1/n1

    """
    It can be checked that the current mean vector has the same values
    in the coordinates (0,5) and (7,4) that the previosly calculated means.
    Hence, it's right to assume that the vector means are correct
    
    print mu0, mu1
    """
   
    # The priors are found:
    p0,p1=100*n0/float(n0+n1),100*n1/float(n0+n1)

    # Finally,  we proceed to find the covariance matrices:
    inter01,inter02,inter1,inter2=[],[],np.zeros((s,s)),np.zeros((s,s))
    for x in range(len(x_train)):
        if y_train_binary[x]==0:
            inter01=x_train[x]
            for i in range(s):
                for j in range(s):
                    inter1[i][j]=(inter01[j]-mu0[j])*(inter01[i]-mu0[i])
            covmat0=np.sum([covmat0,inter1], axis=0)
        else:
            inter02=x_train[x]
            for i in range(s):
                for j in range(s):
                    inter2[i][j]=(inter02[j]-mu1[j])*(inter02[i]-mu1[i])
            covmat1=np.sum([covmat1,inter2], axis=0)

    covmat1=covmat1/n1 
    covmat0=covmat0/n0  
    
    return mu0, mu1, covmat0, covmat1, p0, p1
    
def perform_qda(mu0, mu1, covmat0, covmat1, p0, p1,x_test):
    # The difference between the two quadratic discriminant functions is found,
    # and the label of each sample assigned.
    priors_and_cov,qda_pred=np.log(p0/p1)-0.5*np.linalg.slogdet(covmat0)[1]+0.5*np.linalg.slogdet(covmat1)[1],[]
    
    for x in x_test:
        f0=-0.5*(np.dot(np.dot(x-mu0,np.linalg.inv(covmat0)),x-mu0))
        f1=0.5*(np.dot(np.dot(x-mu1,np.linalg.inv(covmat1)),x-mu1))
        if (f0+f1+priors_and_cov)>0:
            qda_pred.append(1)
        else:
            qda_pred.append(7)
   
    return qda_pred
    
def decision_boundary_qda(mu0, mu1, covmat0, covmat1, p0, p1):
    priors_and_cov,b=np.log(p0/p1)-0.5*(np.linalg.slogdet(covmat0)[1])+0.5*(np.linalg.slogdet(covmat1)[1]),0
    
    boundary,a=np.empty((25,2)),0
    for x in range(25):
        for y in range(4,10):
            f0=-0.5*(np.dot(np.dot([x,y]-mu0,np.linalg.inv(covmat0)),[x,y]-mu0))
            f1=0.5*(np.dot(np.dot([x,y]-mu1,np.linalg.inv(covmat1)),[x,y]-mu1))
            if (f0+f1+priors_and_cov)>0:
                if (f0+f1+priors_and_cov+b)>0:
                    boundary[a]=[x,y-1]
                    a+=1
                    break
                else:
                    boundary[a]=[x,y]
                    a+=1
                    break
            b=f0+f1+priors_and_cov
    
    return boundary
    
def perform_lda(mu0, mu1, covmat0, covmat1, p0, p1,x_test):
    # The difference between the two linear discriminant functions is found,
    # and the label of each sample is assigned.
    priors,qda_pred=np.log(p0/p1),[]
    for x in x_test:
        f0=np.dot(np.dot(x,np.linalg.inv(covmat0)),mu0)-0.5*np.dot(np.dot(mu0,np.linalg.inv(covmat0)),mu0)
        f1=np.dot(np.dot(x,np.linalg.inv(covmat1)),mu1)-0.5*np.dot(np.dot(mu1,np.linalg.inv(covmat1)),mu1)
        if (f0-f1+priors)>0:
            qda_pred.append(1)
        else:
            qda_pred.append(7)
   
    return qda_pred    

def decision_boundary_lda(mu0, mu1, covmat0, covmat1, p0, p1):
    priors,b=np.log(p0/p1),0
    boundary,a=np.empty((17,2)),0
    for x in range(17):
        for y in range(0,17):
            f0=np.dot(np.dot([x,y],np.linalg.inv(covmat0)),mu0)-0.5*np.dot(np.dot(mu0,np.linalg.inv(covmat0)),mu0)
            f1=np.dot(np.dot([x,y],np.linalg.inv(covmat1)),mu1)-0.5*np.dot(np.dot(mu1,np.linalg.inv(covmat1)),mu1)
        if (f0-f1+priors)>0:
                if (f0-f1+priors+b)>0:
                    boundary[a]=[x,y-1]
                    a+=1
                    break
                else:
                    boundary[a]=[x,y]
                    a+=1
                    break
        b=f0-f1+priors
    
    return boundary
    
def main():
    """ All images have been hidden inside a comment in order to speed up
    the code"""
    digits=load_digits(8)
    print digits.keys()

    data = digits["data"]
    target = digits ["target"]
     
    # This function tell us the data contains 1797 images formed by 8*8 pixels.    
    print "\nShape:\n",data.shape

    # With the following loop, the numbers 1 and 7 are filtered: 
    target_1_7=[]
    data_1_7=np.empty((361,64))
    a=0
    for x in range(len(data)):
        if target[x]==1 or target[x]==7:
            data_1_7[a]=data[x]
            a+=1
            target_1_7.append(target[x])
    
    # The data set is splitted into training and test set:
    x_train,x_test,y_train,y_test=\
        cross_validation.train_test_split(data_1_7,target_1_7,
                                          test_size=0.4,random_state=0)
                                          
    # I've chosen these two pixels [ (0,5) and (7,4) ] since they present the 
    # biggest differences between the two digits
    x_train_dr=Dimension_Reduction(x_train)     
    x1,x2,y1,y2=[],[],[],[]
    for x in range(len(x_train_dr)):
        if y_train[x]==1:
            x1.append(x_train_dr[x][0])
            y1.append(x_train_dr[x][1])
        else:
            x2.append(x_train_dr[x][0])
            y2.append(x_train_dr[x][1])
    """
    plt.title ('Scatter Plot ')
    plt.xlabel (' X axis ')
    plt.ylabel (' Y axis ')
    # b: blue , g: green , r: red , c: cyan ,
    # m: magenta , y: yellow , k: black , w: white
    plt.scatter (x1 ,y1 , marker ="x", c="r", s= 50)
    plt.scatter (x2 ,y2 , marker ="o", c="b", s= 13)
    plt.show ()
    """
    # In the figure it can be checked that the samples don't overlap too much,
    # so we keep the selected features
    
    """
    In the nearest mean classifier we find the mean of the feature values 
    of each class (2 classes and 2 features in this case) and predict in 
    which class a testpoint belongs (by choosing the nearest mean).
    
    The exercise says that the function should look like: 
    testy = nm(trainingx, trainingy, testx), but to save work I'll pass
    the values x1,x2,y1,y2 that I have found before, because I would have to
    do exactly the same in this function
    """
    x_test_dr=Dimension_Reduction(x_test) 
    prediction,std_1,std_7 = Nearest_Mean(x1,y1,x2,y2,x_test_dr)
    
    error=0
    for x in range(len(x_test_dr)):
        if y_test[x] != prediction[x]:
            error+=1
    print "\nNearest Mean. Number of mislabeled points:",error," (",(100*error/len(y_test)),"%)"
    # Only 11 errors out of a total of 145 samples.
 
    # Here, we proceed with QDA. 0 and 1 stand for numbers 1 and 7 respectively.
    y_train_binary=[]
    for x in y_train:
        if x==1:
            y_train_binary.append(0)
        else:
            y_train_binary.append(1)
    
    # the covariance matrix, means and priors are calculated for the data set
    # of numbers 1 and 7 with 64 features
    mu0, mu1, covmat0, covmat1, p0, p1 = compute_qda(x_train, y_train_binary)
    """
    Both determinants of the covariance matrix are 0, because the first pixel
    of the numbers 1 and 7 has always the value 0, implying that the covariance
    matrix will have a row and a column with all zeros, thus, as the matrix has a
    linearly dependent row the determinant is 0.
    Since the determinants of the covariance matrices are 0, it hasn't been
    possible to perform qda with the 64 features. However, it can be performed
    if the reduced feature space is used instead (its covariance matrix determinant 
    is not 0):
    """
    mu0, mu1, covmat0, covmat1, p0, p1 = compute_qda(x_train_dr, y_train_binary)
    qda_prediction_test = perform_qda(mu0, mu1, covmat0, covmat1, p0, p1,x_test_dr)
    
    error=0
    for x in range(len(x_test)):
        if y_test[x] != qda_prediction_test[x]:
            error+=1
    print "\nQDA. Number of mislabeled points (test set):",error," (",(100*error/len(y_test)),"%)"
    # Only 14 errors out of a total of 145 samples.
    
    qda_prediction_train = perform_qda(mu0, mu1, covmat0, covmat1, p0, p1,x_train_dr)
    
    error=0
    x1_correct,x2_correct,y1_correct,y2_correct=[],[],[],[]
    x1_incorrect,x2_incorrect,y1_incorrect,y2_incorrect=[],[],[],[]            
    for x in range(len(x_train_dr)):
        if qda_prediction_train[x] ==1:
            if y_train[x]==qda_prediction_train[x]:
                x1_correct.append(x_train_dr[x][0])
                y1_correct.append(x_train_dr[x][1])
            else:
                x1_incorrect.append(x_train_dr[x][0])
                y1_incorrect.append(x_train_dr[x][1])
                error+=1
        else:
            if y_train[x]==qda_prediction_train[x]:
                x2_correct.append(x_train_dr[x][0])
                y2_correct.append(x_train_dr[x][1])
            else:
                x2_incorrect.append(x_train_dr[x][0])
                y2_incorrect.append(x_train_dr[x][1])
                error+=1
           
    print "\nQDA. Number of mislabeled points (train set):",error," (",(100*error/len(y_train)),"%)"
    # Only 13 errors out of a total of 216 samples.
        
    boundary=decision_boundary_qda(mu0, mu1, covmat0, covmat1, p0, p1)
    
    b1,b2=[],[]
    for x in range(len(boundary)):
        b1.append(boundary[x][0])
        b2.append(boundary[x][1])
    """
    f2=plt.figure ('Scatter Plot ')
    plt.xlabel ('Number 1: cross (correct/incorrect: cyan/blue ), and 7: circle (correct/incorrect: yellow/red ) ')
    plt.ylabel (' Y axis ')
    # b: blue , g: green , r: red , c: cyan ,
    # m: magenta , y: yellow , k: black , w: white
    plt.scatter (x1_correct ,y1_correct , marker ="x", c="c", s= 25)
    plt.scatter (x1_incorrect ,y1_incorrect , marker ="x", c="b", s= 25)
    plt.scatter (b1 ,b2 , marker ="_", c="k", s= 20)
    plt.scatter (x2_correct ,y2_correct , marker ="o", c="y", s= 30)
    plt.scatter (x2_incorrect ,y2_incorrect , marker ="o", c="r", s= 30)
    f2.show()
    """
    lda_prediction_train = perform_lda(mu0, mu1, covmat0, covmat1, p0, p1,x_train_dr)
    
    error=0
    x1_correct,x2_correct,y1_correct,y2_correct=[],[],[],[]
    x1_incorrect,x2_incorrect,y1_incorrect,y2_incorrect=[],[],[],[]            
    for x in range(len(x_train_dr)):
        if lda_prediction_train[x] ==1:
            if y_train[x]==lda_prediction_train[x]:
                x1_correct.append(x_train_dr[x][0])
                y1_correct.append(x_train_dr[x][1])
            else:
                x1_incorrect.append(x_train_dr[x][0])
                y1_incorrect.append(x_train_dr[x][1])
                error+=1
        else:
            if y_train[x]==lda_prediction_train[x]:
                x2_correct.append(x_train_dr[x][0])
                y2_correct.append(x_train_dr[x][1])
            else:
                x2_incorrect.append(x_train_dr[x][0])
                y2_incorrect.append(x_train_dr[x][1])
                error+=1
    print "\nLDA. Number of mislabeled points (train set):",error," (",(100*error/len(y_train)),"%)"
    # Only 13 errors out of a total of 216 samples.
    boundary=decision_boundary_lda(mu0, mu1, covmat0, covmat1, p0, p1)
    
    
    b1,b2=[],[]
    for x in range(len(boundary)):
        b1.append(boundary[x][0])
        b2.append(boundary[x][1])
    """
    f3=plt.figure ('Scatter Plot ')
    plt.xlabel ('Number 1: cross (correct/incorrect: cyan/blue ), and 7: circle (correct/incorrect: yellow/red ) ')
    plt.ylabel (' Y axis ')
    # b: blue , g: green , r: red , c: cyan ,
    # m: magenta , y: yellow , k: black , w: white
    plt.scatter (x1_correct ,y1_correct , marker ="x", c="c", s= 25)
    plt.scatter (x1_incorrect ,y1_incorrect , marker ="x", c="b", s= 25)
    plt.scatter (b1 ,b2 , marker ="+", c="k", s= 20)
    plt.scatter (x2_correct ,y2_correct , marker ="o", c="y", s= 30)
    plt.scatter (x2_incorrect ,y2_incorrect , marker ="o", c="r", s= 30)
    f3.show ()
    """
    a=0
    for x in range(len(x_train_dr)):
        if y_train[x]==1:
            x_train_dr[a]=(x_train_dr[x]-mu0)/std_1
            a+=1
        else:
            x_train_dr[a]=(x_train_dr[x]-mu1)/std_7
            a+=1
            
    #lda_prediction_train = perform_lda(mu0, mu1, covmat0, covmat1, p0, p1,x_train_dr)
    
main()        