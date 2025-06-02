import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def train_classifier(X, y, classifier_type='svm', params=None, cv=5):
    """
    Train a classifier with the given features and labels.
    
    Parameters:
    - X: Feature vectors (n_samples, n_features)
    - y: Labels (n_samples,)
    - classifier_type: Type of classifier to use ('svm', 'knn', 'rf', 'dt', 'nb', 'lr')
    - params: Dictionary of parameters for the classifier
    - cv: Number of cross-validation folds for hyperparameter tuning
    
    Returns:
    - Trained classifier object
    """
    # Set default parameters if not provided
    if params is None:
        params = {}

    # Standardize features before classification
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize classifier based on the classifier type
    if classifier_type == 'svm':
        model = SVC()
        if 'kernel' not in params:
            params['kernel'] = ['linear', 'rbf', 'poly']  # Example of parameter grid
        if 'C' not in params:
            params['C'] = [0.1, 1, 10]
        
    elif classifier_type == 'knn':
        model = KNeighborsClassifier()
        if 'n_neighbors' not in params:
            params['n_neighbors'] = [3, 5, 7, 9]  # Example of parameter grid

    elif classifier_type == 'rf':
        model = RandomForestClassifier()
        if 'n_estimators' not in params:
            params['n_estimators'] = [50, 100, 200]
        if 'max_depth' not in params:
            params['max_depth'] = [None, 10, 20]

    elif classifier_type == 'dt':
        model = DecisionTreeClassifier()
        if 'max_depth' not in params:
            params['max_depth'] = [None, 10, 20]
        if 'min_samples_split' not in params:
            params['min_samples_split'] = [2, 5]

    elif classifier_type == 'nb':
        model = GaussianNB()

    elif classifier_type == 'lr':
        model = LogisticRegression(max_iter=1000)
        if 'C' not in params:
            params['C'] = [0.1, 1, 10]

    else:
        raise ValueError(f"Classifier {classifier_type} not supported.")

    # Perform GridSearchCV for hyperparameter tuning
    if len(params) > 0:
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv)
        grid_search.fit(X_scaled, y)
        best_model = grid_search.best_estimator_
        print(f"Best hyperparameters: {grid_search.best_params_}")
    else:
        # No hyperparameter tuning, just fit the model
        best_model = model.fit(X_scaled, y)

    return best_model


def predict(model, X):
    """
    Make predictions using a trained classifier.
    
    Parameters:
    - model: Trained classifier object
    - X: Feature vectors (n_samples, n_features)
    
    Returns:
    - Predicted labels (n_samples,)
    """
    # Standardize features before prediction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return model.predict(X_scaled)


