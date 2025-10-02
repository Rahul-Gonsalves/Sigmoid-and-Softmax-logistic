import os
import matplotlib.pyplot as plt
import numpy as np
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"

def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
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
    print("Plot saved as 'train_features.png' and 'train_features.pdf'")

def main():
    # Load and preprocess data
    print("Loading data...")
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    print(f"Loaded {raw_data.shape[0]} samples")
    
    # Split data
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)
    print(f"Training samples: {raw_train.shape[0]}, Validation samples: {raw_valid.shape[0]}")

    # Preprocess features
    print("Extracting features...")
    train_X_all = prepare_X(raw_train)
    print(f"Feature matrix shape: {train_X_all.shape}")
    
    # Preprocess labels 
    train_y_all, train_idx = prepare_y(label_train)
    print(f"Binary classification indices found: {len(train_idx[0])} samples")

    # Filter for binary case (classes 1 and 2 only)
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    
    # Use first 1350 samples as specified
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    
    # Convert labels from {1, 2} to {+1, -1}
    print("Converting labels...")
    print(f"Original labels: {np.unique(train_y)}")
    train_y[train_y == 1] = 1   # Keep class 1 as +1
    train_y[train_y == 2] = -1  # Convert class 2 to -1
    print(f"Converted labels: {np.unique(train_y)}")
    print(f"Class distribution: +1: {np.sum(train_y == 1)}, -1: {np.sum(train_y == -1)}")
    
    # Print feature statistics
    print("\nFeature Statistics:")
    print(f"Feature 1 (Symmetry) - Min: {train_X[:, 1].min():.4f}, Max: {train_X[:, 1].max():.4f}, Mean: {train_X[:, 1].mean():.4f}")
    print(f"Feature 2 (Intensity) - Min: {train_X[:, 2].min():.4f}, Max: {train_X[:, 2].max():.4f}, Mean: {train_X[:, 2].mean():.4f}")
    
    # Visualize features (exclude bias term - columns 1 and 2 are symmetry and intensity)
    print("\nGenerating visualization...")
    visualize_features(train_X[:, 1:3], train_y)
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()