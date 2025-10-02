#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        self.W = None 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        
        # Initialize weights if not already done
        if self.W is None:
            self.W = np.random.normal(0, 0.01, (n_features, self.k))
        
        # Convert labels to one-hot encoding
        # labels: [n_samples] with values 0, 1, 2, ..., k-1
        # one_hot: [n_samples, k]
        one_hot = np.zeros((n_samples, self.k))
        one_hot[np.arange(n_samples), labels.astype(int)] = 1
        
        # Mini-batch Gradient Descent for max_iter epochs
        for epoch in range(self.max_iter):
            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)
            
            # Process mini-batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Compute gradient for current mini-batch
                batch_gradient = np.zeros((n_features, self.k))
                
                for i in batch_indices:
                    gradient_i = self._gradient(X[i], one_hot[i])
                    batch_gradient += gradient_i
                
                # Average gradient over mini-batch
                avg_batch_gradient = batch_gradient / len(batch_indices)
                
                # Update weights: W = W - learning_rate * gradient
                self.W = self.W - self.learning_rate * avg_batch_gradient
        
        return self
        ### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector of shape [k,].

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        # Compute linear predictions: z = W^T x
        # W is shape [n_features, k], x is shape [n_features]
        # z is shape [k]
        z = np.dot(self.W.T, _x)  # [k,]
        
        # Compute softmax probabilities
        p = self.softmax(z)  # [k,]
        
        # Gradient for multiclass cross-entropy: ∇E(W) = x * (p - y)^T
        # _x is [n_features], (p - _y) is [k]
        # Result should be [n_features, k]
        gradient_diff = p - _y  # [k,] - [k,] = [k,]
        
        # Outer product: x[:, None] * gradient_diff[None, :]
        _g = np.outer(_x, gradient_diff)  # [n_features, k]
        
        return _g
        ### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

        ### YOUR CODE HERE
        # Numerical stability: subtract max to prevent overflow
        # For vector x, softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        
        # Compute exponentials
        exp_x = np.exp(x_shifted)
        
        # Compute softmax: exp(x_i) / sum(exp(x_j))
        softmax_probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        return softmax_probs
        ### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        ### YOUR CODE HERE
        # Compute linear predictions: Z = X * W
        # X: [n_samples, n_features], W: [n_features, k]
        # Z: [n_samples, k]
        Z = np.dot(X, self.W)
        
        # Compute softmax probabilities for each sample
        # Apply softmax row-wise
        probabilities = self.softmax(Z)  # [n_samples, k]
        
        # Predict class with highest probability
        preds = np.argmax(probabilities, axis=1)  # [n_samples,]
        
        return preds
        ### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        ### YOUR CODE HERE
        # Get predictions for all samples
        predictions = self.predict(X)
        
        # Compute accuracy: fraction of correct predictions
        correct_predictions = (predictions == labels.astype(int))
        accuracy = np.mean(correct_predictions)
        
        return accuracy
        ### END YOUR CODE

