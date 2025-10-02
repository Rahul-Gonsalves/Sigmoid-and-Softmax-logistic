import os
import matplotlib.pyplot as plt
import numpy as np
from DataReader import *
from LogisticRegression import logistic_regression

def main():
    data_dir = "../data/"
    train_filename = "training.npz"
    test_filename = "test.npz"
    
    print("="*70)
    print("LOGISTIC REGRESSION - BINARY CLASSIFICATION TESTING")
    print("="*70)
    
    # ============================================================================
    # TRAINING PHASE
    # ============================================================================
    print("\n1. TRAINING PHASE")
    print("-"*50)
    
    # Load and preprocess training data
    print("Loading training data...")
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
    print(f"Training class distribution: +1: {np.sum(train_y == 1)}, -1: {np.sum(train_y == -1)}")
    
    # Train the best model (BGD showed most stable results)
    print("\nTraining Logistic Regression (BGD)...")
    best_model = logistic_regression(learning_rate=0.5, max_iter=100)
    best_model.fit_BGD(train_X, train_y)
    
    train_accuracy = best_model.score(train_X, train_y)
    print(f"Training completed!")
    print(f"Final weights: {best_model.get_params()}")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # ============================================================================
    # TESTING PHASE
    # ============================================================================
    print(f"\n2. TESTING PHASE")
    print("-"*50)
    
    # Load test data
    print("Loading test data...")
    test_raw_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    print(f"Test data loaded: {test_raw_data.shape[0]} samples")
    
    # Preprocess test features (same pipeline as training)
    print("Preprocessing test features...")
    test_X_all = prepare_X(test_raw_data)
    
    # Preprocess test labels (same pipeline as training)
    test_y_all, test_idx = prepare_y(test_labels)
    print(f"Test data - Total samples: {test_X_all.shape[0]}")
    print(f"Test data - Binary classification samples (classes 1&2): {len(test_idx[0])}")
    
    # Filter for binary case (classes 1 and 2 only)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    
    # Convert test labels from {1, 2} to {+1, -1}
    print("Converting test labels...")
    print(f"Original test labels: {np.unique(test_y)}")
    test_y[test_y == 1] = 1   # Keep class 1 as +1
    test_y[test_y == 2] = -1  # Convert class 2 to -1
    print(f"Converted test labels: {np.unique(test_y)}")
    print(f"Test class distribution: +1: {np.sum(test_y == 1)}, -1: {np.sum(test_y == -1)}")
    
    # ============================================================================
    # MODEL EVALUATION
    # ============================================================================
    print(f"\n3. MODEL EVALUATION")
    print("-"*50)
    
    # Get predictions and probabilities
    test_predictions = best_model.predict(test_X)
    test_probabilities = best_model.predict_proba(test_X)
    test_accuracy = best_model.score(test_X, test_y)
    
    # Detailed results
    correct_predictions = np.sum(test_predictions == test_y)
    
    # Per-class accuracy
    class_pos_mask = (test_y == 1)
    class_neg_mask = (test_y == -1)
    
    pos_correct = np.sum((test_predictions == 1) & (test_y == 1))
    neg_correct = np.sum((test_predictions == -1) & (test_y == -1))
    
    pos_total = np.sum(test_y == 1)
    neg_total = np.sum(test_y == -1)
    
    pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0
    neg_accuracy = neg_correct / neg_total if neg_total > 0 else 0
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    print(f"\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"Model: Logistic Regression (Batch Gradient Descent)")
    print(f"Learning Rate: {best_model.learning_rate}")
    print(f"Max Iterations: {best_model.max_iter}")
    print(f"Final Weights: {best_model.get_params()}")
    print()
    print(f"Training Set Size: {len(train_y)} samples")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print()
    print(f"Test Set Size: {len(test_y)} samples")
    print(f"TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    print(f"Detailed Test Results:")
    print(f"  - Correct Predictions: {correct_predictions}/{len(test_y)}")
    print(f"  - Class +1 (Original Class 1): {pos_correct}/{pos_total} = {pos_accuracy:.4f} ({pos_accuracy*100:.2f}%)")
    print(f"  - Class -1 (Original Class 2): {neg_correct}/{neg_total} = {neg_accuracy:.4f} ({neg_accuracy*100:.2f}%)")
    print()
    
    # Model interpretation
    w0, w1, w2 = best_model.get_params()
    print(f"Model Interpretation:")
    print(f"  - Bias term (w0): {w0:.4f}")
    print(f"  - Symmetry weight (w1): {w1:.4f}")
    print(f"  - Intensity weight (w2): {w2:.4f}")
    print(f"  - Decision boundary: {w0:.4f} + {w1:.4f}*symmetry + {w2:.4f}*intensity = 0")
    
    print("\n" + "="*70)
    print(f"*** FINAL TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%) ***")
    print("="*70)
    
    # Sample predictions for verification
    print(f"\nSample Predictions (first 10 test samples):")
    print("Index | True | Pred | Prob(+1) | Correct")
    print("-" * 45)
    for i in range(min(10, len(test_y))):
        prob_pos = test_probabilities[i, 1]  # Probability of class +1
        is_correct = "✓" if test_predictions[i] == test_y[i] else "✗"
        print(f"{i:5d} | {test_y[i]:4.0f} | {test_predictions[i]:4.0f} | {prob_pos:8.4f} | {is_correct}")

if __name__ == "__main__":
    main()