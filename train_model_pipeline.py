import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix


def load_and_preprocess_data(file_path: str):
    data = pd.read_excel(file_path, index_col=0)
    data = data.iloc[:, 3:-1].fillna(0)
    labels = np.array(data.pop('target'))
    features = np.array(data)
    return features, labels


def scale_features(train_features, val_features):
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    return np.clip(train_features, -5, 5), np.clip(val_features, -5, 5)


def compute_class_weights(labels):
    neg, pos = np.bincount(labels)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    return {0: weight_for_0, 1: weight_for_1}


def define_model(model_type='logistic', class_weights=None):
    if model_type == 'logistic':
        return LogisticRegression(solver='lbfgs', class_weight=class_weights)
    elif model_type == 'random_forest':
        return RandomForestClassifier(class_weight=class_weights)
    elif model_type == 'xgboost':
        return XGBClassifier(scale_pos_weight=class_weights[1] / class_weights[0], tree_method='gpu_hist')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def hyperparameter_tuning(model, param_grid, train_features, train_labels):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    grid_result = grid_search.fit(train_features, train_labels)
    print(f"Best Score: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
    return grid_result


def evaluate_model(grid_result, val_features, val_labels):
    y_pred = grid_result.predict(val_features)
    cm = confusion_matrix(val_labels, y_pred)
    print("Confusion Matrix:\n", cm)


def main(file_path, model_type='logistic'):
    # Load and preprocess data
    features, labels = load_and_preprocess_data(file_path)
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2,
                                                                              random_state=42)
    train_features, val_features = scale_features(train_features, val_features)

    # Define class weights
    class_weights = compute_class_weights(train_labels)

    # Define and tune model
    model = define_model(model_type, class_weights)

    # Define parameter grids for each model
    if model_type == 'logistic':
        param_grid = {'class_weight': [{0: 1, 1: weight} for weight in [10, 25, 30, 35, 40, 45, 50]]}
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'class_weight': [{0: 1, 1: weight} for weight in [10, 25, 30]]
        }
    elif model_type == 'xgboost':
        param_grid = {
            'eta': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'scale_pos_weight': [class_weights[1] / class_weights[0]],
            'gamma': [0.5, 1, 1.5],
            'subsample': [0.6, 0.8, 1.0]
        }
    else:
        param_grid = {}

    if param_grid:
        grid_result = hyperparameter_tuning(model, param_grid, train_features, train_labels)
        evaluate_model(grid_result, val_features, val_labels)
    else:
        model.fit(train_features, train_labels)
        evaluate_model(model, val_features, val_labels)


if __name__ == "__main__":
    # Specify the dataset path and model type ('logistic', 'random_forest', or 'xgboost')
    main(file_path='./dataset/data_test.xlsx', model_type='xgboost')
