import os
import matplotlib.pyplot as plt
import numpy as np
from DataReader import *
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass

def compare_sigmoid_vs_softmax():
    """
    Experimental comparison between Sigmoid and Softmax logistic regression
    using data from classes 1 and 2.
    """
    
    data_dir = "../data/"
    train_filename = "training.npz"
    
    print("="*80)
    print("EXPERIMENTAL COMPARISON: SIGMOID vs SOFTMAX LOGISTIC REGRESSION")
    print("="*80)
    
    # ============================================================================
    # DATA PREPARATION
    # ============================================================================
    print("\n1. DATA PREPARATION")
    print("-"*50)
    
    # Load and preprocess data
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)
    
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)
    
    # Use data from classes 1 and 2 only (as specified)
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    
    print(f"Training data shape: {train_X.shape}")
    print(f"Validation data shape: {valid_X.shape}")
    print(f"Original training labels: {np.unique(train_y)}")
    print(f"Original validation labels: {np.unique(valid_y)}")
    
    # ============================================================================
    # SOFTMAX CLASSIFIER (k=2, labels 0,1)
    # ============================================================================
    print(f"\n2. SOFTMAX LOGISTIC REGRESSION (k=2)")
    print("-"*50)
    
    # Prepare labels for softmax: class 1 -> 1, class 2 -> 0
    train_y_softmax = train_y.copy()
    valid_y_softmax = valid_y.copy()
    train_y_softmax[train_y == 2] = 0  # Class 2 becomes 0
    valid_y_softmax[valid_y == 2] = 0  # Class 2 becomes 0
    
    print(f"Softmax training labels: {np.unique(train_y_softmax)}")
    print(f"Softmax validation labels: {np.unique(valid_y_softmax)}")
    
    # Train softmax classifier
    print("Training Softmax classifier...")
    softmax_model = logistic_regression_multiclass(learning_rate=0.01, max_iter=1000, k=2)
    softmax_model.fit_miniBGD(train_X, train_y_softmax.astype(int), batch_size=10)
    
    # Evaluate softmax
    softmax_train_acc = softmax_model.score(train_X, train_y_softmax)
    softmax_valid_acc = softmax_model.score(valid_X, valid_y_softmax)
    
    print(f"Softmax Results:")
    print(f"  Training Accuracy: {softmax_train_acc:.4f} ({softmax_train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {softmax_valid_acc:.4f} ({softmax_valid_acc*100:.2f}%)")
    
    # ============================================================================
    # SIGMOID CLASSIFIER (labels -1, +1)
    # ============================================================================
    print(f"\n3. SIGMOID LOGISTIC REGRESSION (Binary)")
    print("-"*50)
    
    # Prepare labels for sigmoid: class 1 -> +1, class 2 -> -1
    train_y_sigmoid = train_y.copy()
    valid_y_sigmoid = valid_y.copy()
    train_y_sigmoid[train_y == 1] = 1   # Class 1 becomes +1
    train_y_sigmoid[train_y == 2] = -1  # Class 2 becomes -1
    valid_y_sigmoid[valid_y == 1] = 1   # Class 1 becomes +1
    valid_y_sigmoid[valid_y == 2] = -1  # Class 2 becomes -1
    
    print(f"Sigmoid training labels: {np.unique(train_y_sigmoid)}")
    print(f"Sigmoid validation labels: {np.unique(valid_y_sigmoid)}")
    
    # Train sigmoid classifier
    print("Training Sigmoid classifier...")
    sigmoid_model = logistic_regression(learning_rate=0.01, max_iter=1000)
    sigmoid_model.fit_miniBGD(train_X, train_y_sigmoid, batch_size=10)
    
    # Evaluate sigmoid
    sigmoid_train_acc = sigmoid_model.score(train_X, train_y_sigmoid)
    sigmoid_valid_acc = sigmoid_model.score(valid_X, valid_y_sigmoid)
    
    print(f"Sigmoid Results:")
    print(f"  Training Accuracy: {sigmoid_train_acc:.4f} ({sigmoid_train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {sigmoid_valid_acc:.4f} ({sigmoid_valid_acc*100:.2f}%)")
    
    # ============================================================================
    # DETAILED COMPARISON AND ANALYSIS
    # ============================================================================
    print(f"\n4. DETAILED COMPARISON AND ANALYSIS")
    print("="*80)
    
    print(f"ACCURACY COMPARISON:")
    print(f"{'Method':<15} {'Train Acc':<15} {'Valid Acc':<15} {'Difference'}")
    print("-" * 65)
    print(f"{'Softmax':<15} {softmax_train_acc:.4f} ({softmax_train_acc*100:5.1f}%) {softmax_valid_acc:.4f} ({softmax_valid_acc*100:5.1f}%) {abs(softmax_train_acc - softmax_valid_acc):.4f}")
    print(f"{'Sigmoid':<15} {sigmoid_train_acc:.4f} ({sigmoid_train_acc*100:5.1f}%) {sigmoid_valid_acc:.4f} ({sigmoid_valid_acc*100:5.1f}%) {abs(sigmoid_train_acc - sigmoid_valid_acc):.4f}")
    
    accuracy_diff = abs(softmax_valid_acc - sigmoid_valid_acc)
    print(f"\nValidation Accuracy Difference: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
    
    # Weight comparison
    print(f"\nWEIGHT ANALYSIS:")
    print(f"  Softmax weights shape: {softmax_model.W.shape}")
    print(f"  Sigmoid weights shape: {sigmoid_model.W.shape}")
    print(f"  Softmax weights (class 0): {softmax_model.W[:, 0]}")
    print(f"  Softmax weights (class 1): {softmax_model.W[:, 1]}")  
    print(f"  Sigmoid weights:           {sigmoid_model.W}")
    
    # Decision boundary analysis
    print(f"\nDECISION BOUNDARY ANALYSIS:")
    
    # For softmax with 2 classes, the decision boundary is where P(class 0) = P(class 1) = 0.5
    # This happens when w0^T x = w1^T x, or (w1 - w0)^T x = 0
    softmax_boundary_normal = softmax_model.W[:, 1] - softmax_model.W[:, 0]
    
    # For sigmoid, the decision boundary is where σ(w^T x) = 0.5, which is w^T x = 0
    sigmoid_boundary_normal = sigmoid_model.W
    
    print(f"  Softmax boundary normal: {softmax_boundary_normal}")
    print(f"  Sigmoid boundary normal: {sigmoid_boundary_normal}")
    
    # Normalize both to compare direction
    softmax_norm = softmax_boundary_normal / np.linalg.norm(softmax_boundary_normal)
    sigmoid_norm = sigmoid_boundary_normal / np.linalg.norm(sigmoid_boundary_normal)
    
    # Check if they point in the same or opposite direction
    dot_product = np.dot(softmax_norm, sigmoid_norm)
    print(f"  Normalized dot product: {dot_product:.4f}")
    if abs(dot_product) > 0.9:
        direction = "same" if dot_product > 0 else "opposite"
        print(f"  ✓ Boundaries are aligned ({direction} direction)")
    else:
        print(f"  ⚠ Boundaries are not well aligned")
    
    # Prediction comparison on a sample
    print(f"\nPREDICTION COMPARISON (first 10 samples):")
    print(f"{'Sample':<8} {'True':<6} {'Softmax':<8} {'Sigmoid':<8} {'Match'}")
    print("-" * 45)
    
    agreement_count = 0
    total_compared = min(50, len(train_X))
    
    for i in range(total_compared):
        true_label = int(train_y[i])
        
        # Softmax prediction (convert back to original labels)
        softmax_pred_01 = softmax_model.predict(train_X[i:i+1])[0]
        softmax_pred_orig = 1 if softmax_pred_01 == 1 else 2
        
        # Sigmoid prediction (convert back to original labels)  
        sigmoid_pred_pm1 = sigmoid_model.predict(train_X[i:i+1])[0]
        sigmoid_pred_orig = 1 if sigmoid_pred_pm1 == 1 else 2
        
        match = softmax_pred_orig == sigmoid_pred_orig
        if match:
            agreement_count += 1
        
        if i < 10:  # Only print first 10
            match_symbol = "✓" if match else "✗"
            print(f"{i+1:<8} {true_label:<6} {softmax_pred_orig:<8} {sigmoid_pred_orig:<8} {match_symbol}")
    
    agreement_rate = agreement_count / total_compared
    print(f"\nPrediction Agreement Rate: {agreement_count}/{total_compared} = {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
    
    # ============================================================================
    # PROBABILISTIC OUTPUT COMPARISON
    # ============================================================================
    print(f"\n5. PROBABILISTIC OUTPUT COMPARISON")
    print("-" * 50)
    
    # Compare probabilistic outputs for first few samples
    print(f"{'Sample':<8} {'Softmax P(1)':<12} {'Sigmoid P(1)':<12} {'Difference'}")
    print("-" * 50)
    
    prob_diffs = []
    for i in range(min(10, len(train_X))):
        # Softmax probability for class 1 (manually compute)
        z_softmax = np.dot(train_X[i:i+1], softmax_model.W)  # [1, 2]
        softmax_probs = softmax_model.softmax(z_softmax)     # [1, 2]
        softmax_p1 = float(softmax_probs[0, 1])  # Probability of class 1
        
        # Sigmoid probability for class 1 (using predict_proba)
        sigmoid_prob_raw = sigmoid_model.predict_proba(train_X[i:i+1])  # [1, 2]
        sigmoid_p1 = float(sigmoid_prob_raw[0, 1])  # P(y=+1), which is class 1
        
        prob_diff = abs(softmax_p1 - sigmoid_p1)
        prob_diffs.append(prob_diff)
        
        print(f"{i+1:<8} {softmax_p1:<12.4f} {sigmoid_p1:<12.4f} {prob_diff:.4f}")
    
    avg_prob_diff = np.mean(prob_diffs)
    print(f"\nAverage Probability Difference: {avg_prob_diff:.4f}")
    
    # ============================================================================
    # OBSERVATIONS AND INSIGHTS
    # ============================================================================
    print(f"\n6. OBSERVATIONS AND INSIGHTS")
    print("="*80)
    
    print("KEY OBSERVATIONS:")
    print(f"1. THEORETICAL EQUIVALENCE:")
    print(f"   - For binary classification, softmax with k=2 should be equivalent to sigmoid")
    print(f"   - Both model the same decision boundary, just with different parameterizations")
    
    print(f"\n2. ACCURACY COMPARISON:")
    print(f"   - Validation accuracy difference: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
    if accuracy_diff < 0.01:
        print(f"   - ✓ Very similar performance (< 1% difference) - confirms theoretical equivalence")
    elif accuracy_diff < 0.05:
        print(f"   - ✓ Similar performance (< 5% difference) - likely due to optimization differences")
    else:
        print(f"   - ⚠ Significant difference (> 5%) - may indicate implementation or convergence issues")
    
    print(f"\n3. DECISION BOUNDARY ALIGNMENT:")
    if abs(dot_product) > 0.9:
        print(f"   - ✓ Decision boundaries are well-aligned (dot product: {dot_product:.4f})")
        print(f"   - This confirms both methods learned similar decision rules")
    else:
        print(f"   - ⚠ Decision boundaries differ (dot product: {dot_product:.4f})")
        print(f"   - May indicate different local optima or insufficient training")
    
    print(f"\n4. PREDICTION AGREEMENT:")
    print(f"   - Models agree on {agreement_rate*100:.1f}% of predictions")
    if agreement_rate > 0.95:
        print(f"   - ✓ Very high agreement - models are essentially equivalent")
    elif agreement_rate > 0.85:
        print(f"   - ✓ High agreement - models are very similar")
    else:
        print(f"   - ⚠ Lower agreement - may indicate different solutions")
    
    print(f"\n5. PROBABILISTIC OUTPUT:")
    print(f"   - Average probability difference: {avg_prob_diff:.4f}")
    if avg_prob_diff < 0.05:
        print(f"   - ✓ Very similar probability estimates")
    elif avg_prob_diff < 0.1:
        print(f"   - ✓ Reasonably similar probability estimates") 
    else:
        print(f"   - ⚠ Significant probability differences")
    
    print(f"\n6. COMPUTATIONAL ASPECTS:")
    print(f"   - Softmax: {softmax_model.W.shape[0]} × {softmax_model.W.shape[1]} = {np.prod(softmax_model.W.shape)} parameters")
    print(f"   - Sigmoid: {sigmoid_model.W.shape[0]} parameters")
    print(f"   - Softmax requires 2× more parameters but offers extensibility to k>2 classes")
    
    print(f"\n7. PRACTICAL CONCLUSIONS:")
    print(f"   - Both methods should converge to equivalent decision boundaries for binary problems")
    print(f"   - Minor differences are expected due to:")
    print(f"     • Different parameterizations")
    print(f"     • Stochastic optimization effects")  
    print(f"     • Different label encodings")
    print(f"   - For pure binary classification: sigmoid is more efficient")
    print(f"   - For potential multiclass extension: softmax is more flexible")
    
    return {
        'softmax_model': softmax_model,
        'sigmoid_model': sigmoid_model,
        'softmax_acc': softmax_valid_acc,
        'sigmoid_acc': sigmoid_valid_acc,
        'accuracy_difference': accuracy_diff,
        'agreement_rate': agreement_rate,
        'boundary_alignment': dot_product,
        'avg_prob_diff': avg_prob_diff
    }

if __name__ == "__main__":
    results = compare_sigmoid_vs_softmax()