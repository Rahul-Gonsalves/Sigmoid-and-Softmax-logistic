import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None

    def _sigmoid(self, z):
        """Sigmoid function with numerical stability.
        
        Args:
            z: A scalar or array.
            
        Returns:
            Sigmoid of z.
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE
        # Initialize weights if not already done
        if self.W is None:
            self.W = np.random.normal(0, 0.01, n_features)
        
        # Batch Gradient Descent for max_iter epochs
        for epoch in range(self.max_iter):
            # Compute gradient using all samples
            total_gradient = np.zeros(n_features)
            
            # Sum gradients over all training samples
            for i in range(n_samples):
                gradient_i = self._gradient(X[i], y[i])
                total_gradient += gradient_i
            
            # Average gradient over all samples
            avg_gradient = total_gradient / n_samples
            
            # Update weights: W = W - learning_rate * gradient
            self.W = self.W - self.learning_rate * avg_gradient
        ### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        
        # Initialize weights if not already done
        if self.W is None:
            self.W = np.random.normal(0, 0.01, n_features)
        
        # Mini-batch Gradient Descent for max_iter epochs
        for epoch in range(self.max_iter):
            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)
            
            # Process mini-batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Compute gradient for current mini-batch
                batch_gradient = np.zeros(n_features)
                
                for i in batch_indices:
                    gradient_i = self._gradient(X[i], y[i])
                    batch_gradient += gradient_i
                
                # Average gradient over mini-batch
                avg_batch_gradient = batch_gradient / len(batch_indices)
                
                # Update weights: W = W - learning_rate * gradient
                self.W = self.W - self.learning_rate * avg_batch_gradient
        ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        
        # Initialize weights if not already done
        if self.W is None:
            self.W = np.random.normal(0, 0.01, n_features)
        
        # Stochastic Gradient Descent for max_iter epochs
        for epoch in range(self.max_iter):
            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)
            
            # Update weights using one sample at a time
            for i in indices:
                # Compute gradient for single sample
                gradient_i = self._gradient(X[i], y[i])
                
                # Update weights: W = W - learning_rate * gradient
                self.W = self.W - self.learning_rate * gradient_i
        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        # From derivation: ∇E(w) = -y · x · σ(-y · w^T x)
        # Compute w^T x (dot product)
        w_dot_x = np.dot(self.W, _x)
        
        # Compute -y · w^T x
        neg_y_w_dot_x = -_y * w_dot_x
        
        # Compute σ(-y · w^T x)
        sigmoid_value = self._sigmoid(neg_y_w_dot_x)
        
        # Compute gradient: -y · x · σ(-y · w^T x)
        _g = -_y * _x * sigmoid_value
        
        return _g
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        ### YOUR CODE HERE
        n_samples = X.shape[0]
        
        # Compute linear predictions: w^T x for all samples
        linear_preds = np.dot(X, self.W)
        
        # Compute P(y = +1 | x) = σ(w^T x)
        prob_pos = self._sigmoid(linear_preds)
        
        # Compute P(y = -1 | x) = 1 - P(y = +1 | x)
        prob_neg = 1.0 - prob_pos
        
        # Stack probabilities: [P(y = -1), P(y = +1)]
        # Column 0: probability of class -1
        # Column 1: probability of class +1
        preds_proba = np.column_stack((prob_neg, prob_pos))
        
        return preds_proba
        ### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        # Compute linear predictions: w^T x for all samples
        linear_preds = np.dot(X, self.W)
        
        # Apply decision rule: predict +1 if w^T x >= 0, else predict -1
        # This is more efficient than using sigmoid since we only need the sign
        preds = np.where(linear_preds >= 0, 1, -1)
        
        return preds
        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE
        # Get predictions for all samples
        predictions = self.predict(X)
        
        # Compute accuracy: fraction of correct predictions
        correct_predictions = (predictions == y)
        accuracy = np.mean(correct_predictions)
        
        return accuracy
        ### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

