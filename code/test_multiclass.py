import os
import numpy as np
from DataReader import *
from LRM import logistic_regression_multiclass

def test_multiclass_logistic_regression():
    """Test the multiclass logistic regression implementation."""
    
    data_dir = "../data/"
    train_filename = "training.npz"
    test_filename = "test.npz"
    
    print("="*70)
    print("MULTICLASS LOGISTIC REGRESSION (SOFTMAX) TESTING")
    print("="*70)
    
    # ============================================================================
    # TRAINING PHASE
    # ============================================================================
    print("\n1. TRAINING PHASE - All Classes (0, 1, 2)")
    print("-"*50)
    
    # Load and preprocess training data
    print("Loading training data...")
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    # Preprocess features - use ALL data for multiclass
    train_X_all = prepare_X(raw_train)
    train_y_all, _ = prepare_y(label_train)
    
    print(f"Training data shape: {train_X_all.shape}")
    print(f"Training labels - unique classes: {np.unique(train_y_all)}")
    print(f"Class distribution:")
    for cls in [0, 1, 2]:
        count = np.sum(train_y_all == cls)
        print(f"  Class {cls}: {count} samples ({count/len(train_y_all)*100:.1f}%)")
    
    # Train multiclass logistic regression
    print(f"\nTraining Multiclass Logistic Regression...")
    model = logistic_regression_multiclass(learning_rate=0.01, max_iter=100, k=3)
    model.fit_miniBGD(train_X_all, train_y_all, batch_size=10)
    
    train_accuracy = model.score(train_X_all, train_y_all)
    print(f"Training completed!")
    print(f"Weights shape: {model.get_params().shape}")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # ============================================================================
    # TESTING PHASE
    # ============================================================================
    print(f"\n2. TESTING PHASE - All Classes (0, 1, 2)")
    print("-"*50)
    
    # Load test data
    print("Loading test data...")
    test_raw_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    print(f"Test data loaded: {test_raw_data.shape[0]} samples")
    
    # Preprocess test data
    test_X_all = prepare_X(test_raw_data)
    test_y_all, _ = prepare_y(test_labels)
    
    print(f"Test data shape: {test_X_all.shape}")
    print(f"Test labels - unique classes: {np.unique(test_y_all)}")
    print(f"Test class distribution:")
    for cls in [0, 1, 2]:
        count = np.sum(test_y_all == cls)
        print(f"  Class {cls}: {count} samples ({count/len(test_y_all)*100:.1f}%)")
    
    # ============================================================================
    # MODEL EVALUATION
    # ============================================================================
    print(f"\n3. MODEL EVALUATION")
    print("-"*50)
    
    # Get predictions
    test_predictions = model.predict(test_X_all)
    test_accuracy = model.score(test_X_all, test_y_all)
    
    # Per-class accuracy
    correct_predictions = np.sum(test_predictions == test_y_all)
    
    print(f"Overall Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Correct Predictions: {correct_predictions}/{len(test_y_all)}")
    
    # Detailed per-class analysis
    print(f"\nPer-class Test Accuracy:")
    for cls in [0, 1, 2]:
        class_mask = (test_y_all == cls)
        if np.sum(class_mask) > 0:
            class_correct = np.sum((test_predictions == cls) & (test_y_all == cls))
            class_total = np.sum(class_mask)
            class_accuracy = class_correct / class_total
            print(f"  Class {cls}: {class_correct}/{class_total} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Confusion matrix-like analysis
    print(f"\nPrediction Analysis:")
    print("True\\Pred", end="")
    for pred_cls in [0, 1, 2]:
        print(f"\tCls{pred_cls}", end="")
    print()
    
    for true_cls in [0, 1, 2]:
        print(f"Class {true_cls}", end="")
        for pred_cls in [0, 1, 2]:
            count = np.sum((test_y_all == true_cls) & (test_predictions == pred_cls))
            print(f"\t{count}", end="")
        print()
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    print(f"\n" + "="*70)
    print("MULTICLASS LOGISTIC REGRESSION RESULTS")
    print("="*70)
    print(f"Model: Softmax Logistic Regression (Mini-batch GD)")
    print(f"Classes: 3 (0, 1, 2)")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Max Iterations: {model.max_iter}")
    print(f"Batch Size: 10")
    print()
    print(f"Training Set Size: {len(train_y_all)} samples")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print()
    print(f"Test Set Size: {len(test_y_all)} samples")
    print(f"TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*70)
    
    # Sample predictions
    print(f"\nSample Predictions (first 10 test samples):")
    print("Index | True | Pred | Correct")
    print("-" * 30)
    for i in range(min(10, len(test_y_all))):
        is_correct = "✓" if test_predictions[i] == test_y_all[i] else "✗"
        print(f"{i:5d} | {test_y_all[i]:4.0f} | {test_predictions[i]:4d} | {is_correct}")
    
    return model, test_accuracy

if __name__ == "__main__":
    model, accuracy = test_multiclass_logistic_regression()