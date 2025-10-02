import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Separate data by class
    class_pos = X[y == 1]  # Positive class (originally class 1)
    class_neg = X[y == -1] # Negative class (originally class 2)
    
    # Create scatter plot with different colors for each class
    plt.scatter(class_pos[:, 0], class_pos[:, 1], c='red', marker='o', 
                label='Class 1 (+1)', alpha=0.7, s=50)
    plt.scatter(class_neg[:, 0], class_neg[:, 1], c='blue', marker='^', 
                label='Class 2 (-1)', alpha=0.7, s=50)
    
    # Add labels and title
    plt.xlabel('Feature 1: Measure of Symmetry', fontsize=12)
    plt.ylabel('Feature 2: Measure of Intensity', fontsize=12)
    plt.title('2D Scatter Plot of Training Features (Classes 1 and 2)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('train_features.png', dpi=300, bbox_inches='tight')
    plt.savefig('train_features.pdf', bbox_inches='tight')
    plt.show()
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    # Create figure and axis
    plt.figure(figsize=(12, 9))
    
    # Separate data by class
    class_pos = X[y == 1]  # Positive class (originally class 1)
    class_neg = X[y == -1] # Negative class (originally class 2)
    
    # Create scatter plot with different colors for each class
    plt.scatter(class_pos[:, 0], class_pos[:, 1], c='red', marker='o', 
                label='Class 1 (+1)', alpha=0.7, s=50)
    plt.scatter(class_neg[:, 0], class_neg[:, 1], c='blue', marker='^', 
                label='Class 2 (-1)', alpha=0.7, s=50)
    
    # Plot decision boundary
    # Decision boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1)/w2
    if len(W) == 3:  # W = [w0 (bias), w1 (symmetry), w2 (intensity)]
        # Get feature ranges for plotting
        x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        
        # Create line points for decision boundary
        x1_line = np.linspace(x1_min, x1_max, 100)
        
        # Decision boundary equation: W[0] + W[1]*x1 + W[2]*x2 = 0
        # Solve for x2: x2 = -(W[0] + W[1]*x1) / W[2]
        if abs(W[2]) > 1e-10:  # Avoid division by zero
            x2_line = -(W[0] + W[1] * x1_line) / W[2]
            
            # Only plot points within the data range
            valid_indices = (x2_line >= x2_min) & (x2_line <= x2_max)
            plt.plot(x1_line[valid_indices], x2_line[valid_indices], 
                    'g-', linewidth=3, label='Decision Boundary')
        else:
            # Vertical line case: W[1]*x1 + W[0] = 0 => x1 = -W[0]/W[1]
            if abs(W[1]) > 1e-10:
                x1_boundary = -W[0] / W[1]
                plt.axvline(x=x1_boundary, color='green', linewidth=3, 
                           label='Decision Boundary')
    
    # Add labels and title
    plt.xlabel('Feature 1: Measure of Symmetry', fontsize=12)
    plt.ylabel('Feature 2: Measure of Intensity', fontsize=12)
    plt.title('Logistic Regression Results: Training Data with Decision Boundary', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    plt.ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    
    # Save the plot
    plt.savefig('train_result_sigmoid.png', dpi=300, bbox_inches='tight')
    plt.savefig('train_result_sigmoid.pdf', bbox_inches='tight')
    plt.show()
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    # Create figure and axis
    plt.figure(figsize=(12, 10))
    
    # Define colors and markers for each class
    colors = ['green', 'red', 'blue']
    markers = ['s', 'o', '^']  # square, circle, triangle
    class_names = ['Class 0', 'Class 1', 'Class 2']
    
    # Separate data by class and plot
    for cls in [0, 1, 2]:
        class_mask = (y == cls)
        class_data = X[class_mask]
        if len(class_data) > 0:
            plt.scatter(class_data[:, 0], class_data[:, 1], 
                       c=colors[cls], marker=markers[cls], 
                       label=class_names[cls], alpha=0.7, s=50)
    
    # Create a mesh to plot decision boundaries
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Create mesh grid
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                           np.linspace(x2_min, x2_max, 300))
    
    # For multiclass, we need to add the bias feature for prediction
    # W is [n_features, k] where n_features = 3 (bias, symmetry, intensity)
    # We need to create mesh points with bias term
    mesh_points = np.c_[np.ones(xx1.ravel().shape[0]),  # bias term = 1
                        xx1.ravel(), xx2.ravel()]  # [n_points, 3]
    
    # Compute softmax predictions for each mesh point
    Z = np.dot(mesh_points, W)  # [n_points, 3]
    
    # Apply softmax
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    # Get predicted class (argmax)
    Z_pred = np.argmax(softmax_probs, axis=1)
    Z_pred = Z_pred.reshape(xx1.shape)
    
    # Plot decision boundaries
    plt.contour(xx1, xx2, Z_pred, levels=[0.5, 1.5], 
                colors=['black'], linestyles=['--'], linewidths=2)
    
    # Add filled contours to show decision regions
    plt.contourf(xx1, xx2, Z_pred, levels=[-0.5, 0.5, 1.5, 2.5], 
                 colors=['lightgreen', 'lightcoral', 'lightblue'], alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Feature 1: Measure of Symmetry', fontsize=12)
    plt.ylabel('Feature 2: Measure of Intensity', fontsize=12)
    plt.title('Multiclass Logistic Regression (Softmax): Training Data with Decision Boundaries', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    
    # Save the plot
    plt.savefig('train_result_softmax.png', dpi=300, bbox_inches='tight')
    plt.savefig('train_result_softmax.pdf', bbox_inches='tight')
    plt.show()
    ### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    ### YOUR CODE HERE
    # Convert labels from {1, 2} to {+1, -1}
    # Class 1 becomes +1 (positive class)
    # Class 2 becomes -1 (negative class)
    train_y[train_y == 1] = 1   # Keep class 1 as +1
    train_y[train_y == 2] = -1  # Convert class 2 to -1
    valid_y[valid_y == 1] = 1   # Keep class 1 as +1  
    valid_y[valid_y == 2] = -1  # Convert class 2 to -1
    ### END YOUR CODE
    data_shape = train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    # For demonstration, let's use the BGD model as the 'best' model
    best_logisticR = logistic_regression(learning_rate=0.5, max_iter=100)
    best_logisticR.fit_BGD(train_X, train_y)
    print("Best model (BGD) - Weights:", best_logisticR.get_params())
    print("Best model (BGD) - Training Accuracy:", best_logisticR.score(train_X, train_y))
    
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE

    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    print("\n" + "="*60)
    print("TESTING PHASE: Evaluating Best Model on Test Data")
    print("="*60)
    
    # Load test data
    print("Loading test data...")
    test_raw_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    print(f"Test data loaded: {test_raw_data.shape[0]} samples")
    
    # Preprocess test features (same as training data)
    print("Preprocessing test features...")
    test_X_all = prepare_X(test_raw_data)
    
    # Preprocess test labels (same as training data)
    test_y_all, test_idx = prepare_y(test_labels)
    print(f"Test data - Total samples: {test_X_all.shape[0]}")
    print(f"Test data - Binary classification samples (classes 1&2): {len(test_idx[0])}")
    
    # Filter for binary case (classes 1 and 2 only) - same as training
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    
    # Convert labels from {1, 2} to {+1, -1} - same as training
    print("Converting test labels...")
    print(f"Original test labels: {np.unique(test_y)}")
    test_y[test_y == 1] = 1   # Keep class 1 as +1
    test_y[test_y == 2] = -1  # Convert class 2 to -1
    print(f"Converted test labels: {np.unique(test_y)}")
    print(f"Test class distribution: +1: {np.sum(test_y == 1)}, -1: {np.sum(test_y == -1)}")
    
    # Evaluate the best model on test data
    print(f"\nTesting best model (BGD) on test data...")
    test_accuracy = best_logisticR.score(test_X, test_y)
    test_predictions = best_logisticR.predict(test_X)
    test_probabilities = best_logisticR.predict_proba(test_X)
    
    # Report detailed results
    print(f"\n" + "-"*50)
    print("TEST RESULTS:")
    print("-"*50)
    print(f"Model: Logistic Regression (Batch Gradient Descent)")
    print(f"Best Model Weights: {best_logisticR.get_params()}")
    print(f"Training Accuracy: {best_logisticR.score(train_X, train_y):.4f} ({best_logisticR.score(train_X, train_y)*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Set Size: {len(test_y)} samples")
    
    # Additional analysis
    correct_predictions = np.sum(test_predictions == test_y)
    print(f"Correct Predictions: {correct_predictions}/{len(test_y)}")
    
    # Per-class accuracy
    class_pos_mask = (test_y == 1)
    class_neg_mask = (test_y == -1)
    
    pos_accuracy = np.mean(test_predictions[class_pos_mask] == test_y[class_pos_mask])
    neg_accuracy = np.mean(test_predictions[class_neg_mask] == test_y[class_neg_mask])
    
    print(f"Class +1 Accuracy: {pos_accuracy:.4f} ({pos_accuracy*100:.2f}%)")
    print(f"Class -1 Accuracy: {neg_accuracy:.4f} ({neg_accuracy*100:.2f}%)")
    
    print(f"\n" + "="*60)
    print("FINAL RESULT: Test Accuracy = {:.4f} ({:.2f}%)".format(test_accuracy, test_accuracy*100))
    print("="*60)
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    # Use the trained multiclass model for visualization
    best_logistic_multi_R = logisticR_classifier_multiclass
    print("Multiclass model (Softmax) - Weights shape:", best_logistic_multi_R.get_params().shape)
    print("Multiclass model (Softmax) - Training Accuracy:", best_logistic_multi_R.score(train_X, train_y))
    
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
	### YOUR CODE HERE

	### END YOUR CODE 

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
