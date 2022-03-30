#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
import numpy as np


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        # TODO
        converge = False
        epoch = 0
        print(epoch)
        num_examples = X.shape[0]
        while not converge:
            epoch += 1
            shuffler = np.random.permutation(len(Y))
            array1_shuffled = X[shuffler]
            array2_shuffled = Y[shuffler]
            for i in range(num_examples//self.batch_size):
                X_batch = X[i * self.batch_size: (i+1) * self.batch_size]
                Y_batch = Y[i * self.batch_size: (i+1) * self.batch_size]
                L = np.zeros((self.n_classes, self.n_features + 1))
                for example_no in range(self.batch_size):
                    for j in range(self.n_classes):
                        if Y_batch[example_no] == j:
                            tmp = X_batch[example_no] @ self.weights.T
                            tmp = softmax(tmp)[j] - 1
                            tmp = tmp * X_batch[example_no]
                            L[j]+= tmp
                        else:
                            tmp = X_batch[example_no] @ self.weights.T
                            tmp = softmax(tmp)[j]
                            tmp = tmp * X_batch[example_no]
                            L[j]+= tmp
                self.weights = self.weights - (self.alpha * L / self.batch_size)
            if epoch != 1 and (np.absolute(self.loss(X,Y) - last_loss)< self.conv_threshold):
                converge = True
            last_loss = self.loss(X,Y) 
        return epoch

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        # TODO
        probabilities = X @ self.weights.T 

        error = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            error[i] = np.log(softmax(probabilities[i])[Y[i]])
        return np.sum(error) / X.shape[0]

    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        # TODO
        predicted = X @ self.weights.T 
        predictions = np.argmax(predicted, axis=1)
        return predictions.T

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        # TODO
        res = X @ self.weights.T 
        print(res)
        ex = np.exp(res)
        sum = np.sum(ex, axis = 1)
        sum = sum.reshape(sum.shape[0], 1)
        probabilities = ex/sum

        predictions = np.argmax(probabilities, axis=1)
        sum = 0
        for i in range(Y.shape[0]):
            if predictions[i] == Y[i]:
                sum += 1
        return sum / Y.shape[0]
