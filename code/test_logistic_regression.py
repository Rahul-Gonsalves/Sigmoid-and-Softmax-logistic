import os
import matplotlib.pyplot as plt
import numpy as np
from DataReader import *
from LogisticRegression import logistic_regression

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
    print("Plot saved as 'train_result_sigmoid.png' and 'train_result_sigmoid.pdf'")

def main():
    data_dir = "../data/"
    train_filename = "training.npz"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    # Preprocess features and labels
    train_X_all = prepare_X(raw_train)
    train_y_all, train_idx = prepare_y(label_train)

    # Filter for binary case (classes 1 and 2 only)
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    
    # Use first 1350 samples as specified
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    
    # Convert labels from {1, 2} to {+1, -1}
    train_y[train_y == 1] = 1   # Keep class 1 as +1
    train_y[train_y == 2] = -1  # Convert class 2 to -1
    
    print(f"Training data shape: {train_X.shape}")
    print(f"Class distribution: +1: {np.sum(train_y == 1)}, -1: {np.sum(train_y == -1)}")
    
    # Test different gradient descent methods
    print("\n" + "="*50)
    print("Testing Logistic Regression Implementation")
    print("="*50)
    
    # Test BGD
    print("\n1. Batch Gradient Descent:")
    logisticR_BGD = logistic_regression(learning_rate=0.5, max_iter=100)
    logisticR_BGD.fit_BGD(train_X, train_y)
    print(f"   Weights: {logisticR_BGD.get_params()}")
    print(f"   Training Accuracy: {logisticR_BGD.score(train_X, train_y):.4f}")
    
    # Test SGD
    print("\n2. Stochastic Gradient Descent:")
    logisticR_SGD = logistic_regression(learning_rate=0.5, max_iter=100)
    logisticR_SGD.fit_SGD(train_X, train_y)
    print(f"   Weights: {logisticR_SGD.get_params()}")
    print(f"   Training Accuracy: {logisticR_SGD.score(train_X, train_y):.4f}")
    
    # Test mini-batch GD
    print("\n3. Mini-batch Gradient Descent (batch_size=10):")
    logisticR_miniBGD = logistic_regression(learning_rate=0.5, max_iter=100)
    logisticR_miniBGD.fit_miniBGD(train_X, train_y, 10)
    print(f"   Weights: {logisticR_miniBGD.get_params()}")
    print(f"   Training Accuracy: {logisticR_miniBGD.score(train_X, train_y):.4f}")
    
    # Use the best performing model for visualization (let's use BGD as it's most stable)
    best_model = logisticR_BGD
    print(f"\n4. Using BGD model for visualization:")
    print(f"   Final weights: {best_model.get_params()}")
    
    # Test prediction functions
    print("\n5. Testing prediction functions:")
    sample_probs = best_model.predict_proba(train_X[:5])
    sample_preds = best_model.predict(train_X[:5])
    print(f"   Sample probabilities (first 5): \n{sample_probs}")
    print(f"   Sample predictions (first 5): {sample_preds}")
    print(f"   True labels (first 5): {train_y[:5]}")
    
    # Visualize results with decision boundary
    print("\n6. Generating visualization...")
    visualize_result(train_X[:, 1:3], train_y, best_model.get_params())
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()