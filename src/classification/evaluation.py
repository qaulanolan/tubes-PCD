from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using standard metrics.
    
    Parameters:
    - model: Trained classifier object
    - X_test: Test feature vectors (n_samples, n_features)
    - y_test: True labels for the test set (n_samples,)
    
    Returns:
    - Dictionary with evaluation metrics (accuracy, precision, recall, f1-score)
    """
    
    try:
        # Predict the labels using the trained model
        y_pred = model.predict(X_test)
        
        # Ensure that the shape of y_pred and y_test are consistent
        if y_pred.shape != y_test.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} does not match y_test shape {y_test.shape}")

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Return a dictionary of evaluation metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
