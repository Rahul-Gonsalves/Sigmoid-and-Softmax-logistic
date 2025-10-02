import os
import matplotlib.pyplot as plt
import numpy as np
from DataReader import *
from LRM import logistic_regression_multiclass

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
    print("Multiclass visualization saved as 'train_result_softmax.png' and 'train_result_softmax.pdf'")

def main():
    data_dir = "../data/"
    train_filename = "training.npz"
    
    print("="*80)
    print("TESTING MULTICLASS LOGISTIC REGRESSION WITH VISUALIZATION")
    print("="*80)
    
    # ------------Data Preprocessing------------
    print("\n1. Data Preprocessing...")
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    # Preprocess features and labels - use ALL classes for multiclass
    train_X_all = prepare_X(raw_train)
    train_y_all, _ = prepare_y(label_train)
    
    print(f"Training data shape: {train_X_all.shape}")
    print(f"Training labels - unique classes: {np.unique(train_y_all)}")
    print(f"Class distribution:")
    for cls in [0, 1, 2]:
        count = np.sum(train_y_all == cls)
        print(f"  Class {cls}: {count} samples ({count/len(train_y_all)*100:.1f}%)")
    
    # ------------Multiclass Logistic Regression Training------------
    print(f"\n2. Training Multiclass Logistic Regression...")
    
    # Use all data from classes 0, 1, 2 for multiclass training
    train_X = train_X_all
    train_y = train_y_all
    
    # Train multiclass logistic regression with mini-batch GD
    print("Training with mini-batch gradient descent...")
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=100, k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    
    training_accuracy = logisticR_classifier_multiclass.score(train_X, train_y)
    weights = logisticR_classifier_multiclass.get_params()
    
    print(f"Training completed!")
    print(f"Weights shape: {weights.shape}")
    print(f"Training accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    
    # ------------Visualization------------
    print(f"\n3. Generating Multiclass Visualization...")
    
    # Visualize the multiclass model results
    # Use features 1 and 2 (exclude bias term for visualization)
    print("Creating 2D visualization with decision boundaries...")
    visualize_result_multi(train_X[:, 1:3], train_y, weights)
    
    # ------------Results Summary------------
    print(f"\n" + "="*80)
    print("MULTICLASS LOGISTIC REGRESSION RESULTS")
    print("="*80)
    print(f"Model: Softmax Logistic Regression")
    print(f"Algorithm: Mini-batch Gradient Descent")
    print(f"Learning Rate: {logisticR_classifier_multiclass.learning_rate}")
    print(f"Max Iterations: {logisticR_classifier_multiclass.max_iter}")
    print(f"Batch Size: 10")
    print(f"Classes: 3 (0, 1, 2)")
    print()
    print(f"Training Set Size: {len(train_y)} samples")
    print(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    print()
    print(f"Weight Matrix Shape: {weights.shape}")
    print(f"Features: [Bias, Symmetry, Intensity]")
    print()
    print("Visualization Files:")
    print("  - train_result_softmax.png (high resolution)")
    print("  - train_result_softmax.pdf (vector format)")
    print("="*80)

if __name__ == "__main__":
    main()