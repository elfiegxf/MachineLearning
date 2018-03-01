# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import scipy
import math
import nltk
import sys
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split


def read_in_data_function(datafile_name,vocabulary):
#This function is for reading txt files in
#It will also strip all the redundant info in the txt file such as punctuation.
    X = []
    Y = []
    f = open(datafile_name,"r")
    lines = f.readlines()
    for i in lines:
        digit = [s for s in i if s.isdigit()]
        Y += [int(digit[-1])]
    f.close()
    Y = numpy.asarray(Y)
    count_vect = CountVectorizer(max_df = 0.85, max_features =  int(vocabulary), stop_words = "english")
    X_train_counts = count_vect.fit_transform(lines)
    tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts).toarray()
    X_binary= binarize(X_train_tf)
    for i in X_binary:
        temp = [int(j) for j in i]
        X += [temp]
    X = numpy.asarray(X)
    return [X,Y]

def split_data(X,Y,train_test_ratio):
    #This function used for spliting data into training and testing data.
    #input:
    #   1) X data:[Ntrain ∗ d] numpy array , the training features.
    #   2) Y labels: [Ntrain] numpy vector, the training labels.
    #   3) test_ratio: float, ratio = test sets/sample sets
    #Output:
    #   X_train,X_test,y_train,y_test: numpy arrays.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = train_test_ratio)
    return [X_train, X_test, y_train, y_test]
    
def train(X_train, y_train, train_opt):
# Train the model with Xtrain, ytrain and train_opt.
# Inputs:
# 1 ) Xtrain : [Ntrain ∗ d] numpy array , the training features.
# 2 ) ytrain : [Ntrain] numpy vector, the training labels.
# 3 ) trainopt : [dict], contains some other parameters.
# Output :
# 1 ) trained model : [list] , contains the trained model.
    if len(X_train) == 0:
        raise ValueError('Non-numeric data found in the file.')
    if len(X_train) != len(y_train):
        raise ValueError('The size of X_train and y_train do not match.')
    m = float(train_opt["smooth_param"])
    d = len(X_train[0])
    l = len(X_train)
    #quick sort, result will be [ 0 0  ... 0 , 1 1 ... 1]
    i = 0
    j = l - 1
    k = i
    while  k < j :
            if y_train[k] == 0:
                i += 1
                k += 1
            elif y_train[k] == 1:
                X_train[k], X_train[j] = X_train[j], X_train[k]
                y_train[k], y_train[j] = y_train[j], y_train[k]
                j -= 1
    neg = numpy.zeros(d,dtype = float)
    pos = numpy.zeros(d,dtype = float)
    neg = numpy.add(neg,X_train[0:k].sum(axis=0))
    pos = numpy.add(pos,X_train[k:l].sum(axis=0))
     #conditional probability of each word
    pos = (pos + m) / ((l - k) + m*d)
    neg = (neg + m) / (k + m*d)
    #append the prior probability to the end
    neg = numpy.append(neg,[1.0*k/l])
    pos = numpy.append(pos,[1.0*(l - k)/l])
    trained_model = numpy.asarray(list(zip(neg,pos)))
    return trained_model


def crossValidation(X_train_data,y_train_data,X_test,y_test,split_ratio,k):
    #This function used for crossValidation.
    #Input:
    # 1 ) X_train_data: [Ntrain ∗ d] numpy array , the training features.
    # 2 ) y_train_data: [Ntrain] numpy vector, the training labels.
    # 3 ) split_ratio : float, cratio = test sets/sample sets.
    # 4 ) k, int, k-cross-validation.
    #Output:
    # 5 ) error_mean: float, mean of k times validation errors.
    #     test_error: float, test error for corresponding training model
    error_vali = numpy.zeros(k,dtype = float)
    for i in range(k):
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_data,y_train_data,test_size = 1-split_ratio)
        vali_model = train(X_train,y_train,{"smooth_param":0.01})
        y_result = test(X_validation,vali_model)
        error_vali[i] = evaluate(y_result,y_validation)
    error_mean = numpy.mean(error_vali,dtype = float) 
    test_error = evaluate(y_test,test(X_test,vali_model))
    return [error_mean,test_error] 

def test( X_test, trained_model):
    #Predict the labels for X_test, given the trained model.
    #Input:
    #  1)X_test: [N_test*d] numpy array, the testing features
    #  2)trained_model: [dict] contains the trained model.
    #Output:
    #  1)y-pred: [N-test] numpy vector, the predicted labels.
    if len(X_test) == 0:
        raise ValueError('Non-numeric data found in the file.')
    if len(X_test[0]) != len(trained_model) - 1:
        raise ValueError('The Number of X_train and trained_model do not match.')
    y_pred = []
    count= 0
    d = len(X_test)
    l = len(X_test[0])
    for i in range(d):
        pos = 0.0
        neg = 0.0
        for j in range(l):
            pos += X_test[i][j] * numpy.log(trained_model[j][1])
            neg += X_test[i][j] * numpy.log(trained_model[j][0])
        pos += numpy.log(trained_model[l][1])
        neg += numpy.log(trained_model[l][0])
        if pos > neg:
            y_pred += [1] 
        elif pos < neg:
            y_pred += [0] 
        else:
            y_pred += [count]
            count ^= 1 
    y_pred = numpy.asarray(y_pred)
    return y_pred


def evaluate(y_test , y_pred):
    #Evaluate between y and y_pre, using the options in eval_opt
    #Input:
    # 1) y_test: [N_test] numpy vector, the testing labels.
    # 2) y_pred: [N_test] numpy vector, the predicted labels.
    #Output:
    # 1) error_rate: a float number in [0,1],the error rate of the prediction
    if len(y_test) == 0:
        raise ValueError('Non-numeric data found in the file.')
    if len(y_test) != len(y_pred):    
        raise ValueError('The Number of X_train and y_train do not match.')
    size = len(y_test)
    error_rate = 1.0*numpy.sum(numpy.absolute(numpy.add(y_test, -1*y_pred)),axis = 0)
    return error_rate/size

def main():

    [X,Y] = read_in_data_function("yelp_labelled.txt",500)
    [X_train, X_test, y_train, y_test] = split_data(X,Y,0.1)

    vali_split_radio = [0.1,0.5,0.8,0.95]
    error = numpy.zeros(len(vali_split_radio),dtype = float)
    test_error =  numpy.zeros(len(vali_split_radio),dtype = float)
    for i in range(len(vali_split_radio)):
        [error[i],test_error[i]] = crossValidation(X_train,y_train,X_test,y_test,vali_split_radio[i],5)

    plt.plot(vali_split_radio,error,color="red",linestyle="-")
    plt.plot(vali_split_radio,test_error)
    plt.ylabel('error_rate')
    plt.xlabel('training data ratio')
    plt.title('cross_validation_mean// red: validation error')
    plt.grid(True)
    plt.savefig('cross_validation_mean_error_rate.png')
     
    m = [0.1, 0.5, 2.5, 12.5]
    errors = numpy.zeros(len(m),dtype = float)
    for i in range(len(m)):
        TraindModel = train(X_train, y_train, {"smooth_param":m[i]})
        test_prediction = test(X_test,TraindModel)
        errors[i] = evaluate(y_test,test_prediction)
    plt.clf()
    plt.plot(m,errors)
    plt.xlabel('smoothing_factor')
    plt.ylabel('test_error_rate')
    plt.title('test_error with different smoothing_factors')
    plt.savefig('test_error_rate.png')

    vocabulary_size = [1000, 500, 250, 125]
    error_voc = numpy.zeros(len(vocabulary_size),dtype = float)
    for i in range(len(vocabulary_size)):
        [X,Y] = read_in_data_function("yelp_labelled.txt",vocabulary_size[i])
        [X_train, X_test, y_train, y_test] = split_data(X,Y,0.1)
        TM = train(X_train, y_train, {"smooth_param":0.5})
        test_p = test(X_test,TM)
        error_voc[i] = evaluate(y_test,test_p)
    plt.clf()
    plt.plot(vocabulary_size,error_voc)
    plt.xlabel('vocabulary')
    plt.ylabel('error_rate')
    plt.title('test_error with different vocabulary sizes')
    plt.savefig('vocabulary.png')
main()