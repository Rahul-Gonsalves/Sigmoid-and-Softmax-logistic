import os
import numpy as np
from DataReader import *
from LogisticRegression import logistic_regression

def compare_gradient_descent_methods():
    """Compare different gradient descent methods on test data."""
    
    data_dir = "../data/"
    train_filename = "training.npz"
    test_filename = "test.npz"
    
    print("="*80)
    print("COMPARING GRADIENT DESCENT METHODS - TEST PERFORMANCE")
    print("="*80)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    
    # Training data
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)
    train_X_all = prepare_X(raw_train)
    train_y_all, train_idx = prepare_y(label_train)
    train_X = train_X_all[train_idx][0:1350]
    train_y = train_y_all[train_idx][0:1350]
    train_y[train_y == 1] = 1
    train_y[train_y == 2] = -1
    
    # Test data
    test_raw_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_raw_data)
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[test_y == 1] = 1
    test_y[test_y == 2] = -1
    
    print(f"Training samples: {len(train_y)}")
    print(f"Test samples: {len(test_y)}")
    
    # Test different methods
    methods = [
        ("Batch Gradient Descent", "fit_BGD", {}),
        ("Stochastic Gradient Descent", "fit_SGD", {}),
        ("Mini-batch GD (batch=10)", "fit_miniBGD", {"batch_size": 10}),
        ("Mini-batch GD (batch=50)", "fit_miniBGD", {"batch_size": 50}),
    ]
    
    results = []
    
    print(f"\n" + "-"*80)
    print("METHOD COMPARISON")
    print("-"*80)
    
    for method_name, method_func, kwargs in methods:
        print(f"\nTesting {method_name}...")
        
        # Train model
        model = logistic_regression(learning_rate=0.5, max_iter=100)
        fit_method = getattr(model, method_func)
        
        if kwargs:
            fit_method(train_X, train_y, **kwargs)
        else:
            fit_method(train_X, train_y)
        
        # Evaluate
        train_acc = model.score(train_X, train_y)
        test_acc = model.score(test_X, test_y)
        weights = model.get_params()
        
        results.append({
            'method': method_name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'weights': weights
        })
        
        print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Weights: [{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}]")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'Train Acc':<12} {'Test Acc':<12} {'Generalization'}")
    print("-" * 80)
    
    best_test_acc = 0
    best_method = ""
    
    for result in results:
        generalization = result['test_acc'] - result['train_acc']
        
        if result['test_acc'] > best_test_acc:
            best_test_acc = result['test_acc']
            best_method = result['method']
        
        print(f"{result['method']:<25} {result['train_acc']:.4f} ({result['train_acc']*100:5.1f}%) "
              f"{result['test_acc']:.4f} ({result['test_acc']*100:5.1f}%) "
              f"{generalization:+.4f}")
    
    print("\n" + "="*80)
    print(f"BEST METHOD: {best_method}")
    print(f"BEST TEST ACCURACY: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = compare_gradient_descent_methods()