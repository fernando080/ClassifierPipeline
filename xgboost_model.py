import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def load_and_preprocess_data(file_path: str):
    """Load and preprocess dataset from an Excel file."""
    data = pd.read_excel(file_path, index_col=0)
    data = data.iloc[:, 3:-1].fillna(0)

    # Display class distribution
    neg, pos = np.bincount(data['target'])
    total = neg + pos
    print(f"Examples:\n    Total: {total}\n    Positive: {pos} ({100 * pos / total:.2f}% of total)\n")

    # Split features and labels
    labels = np.array(data.pop('target'))
    features = np.array(data)

    return features, labels


def scale_features(train_features, val_features):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Clipping values for stability
    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)

    return train_features, val_features


def define_model():
    """Define the XGBClassifier model with GPU support."""
    return XGBClassifier(tree_method='gpu_hist')


def hyperparameter_tuning(model, train_features, train_labels):
    """Perform grid search for hyperparameter tuning on XGBClassifier."""
    param_grid = {
        'eta': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'min_child_weight': [1, 5, 10],
        'scale_pos_weight': [24],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')

    grid_result = grid_search.fit(train_features, train_labels)
    print(f"Best Score: {grid_result.best_score_:.4f} using {grid_result.best_params_}")

    # Display all configurations
    for mean, std, params in zip(grid_result.cv_results_['mean_test_score'],
                                 grid_result.cv_results_['std_test_score'],
                                 grid_result.cv_results_['params']):
        print(f"{mean:.4f} ({std:.4f}) with: {params}")

    return grid_result


def evaluate_model(grid_result, val_features, val_labels):
    """Evaluate the best model found on the validation set."""
    y_pred = grid_result.predict(val_features)

    print("Confusion Matrix:\n", confusion_matrix(val_labels, y_pred))
    print("F1 Score:", f1_score(val_labels, y_pred))
    print("Recall:", recall_score(val_labels, y_pred))
    print("Accuracy:", accuracy_score(val_labels, y_pred))


def main():
    # Load and preprocess data
    train_features, train_labels = load_and_preprocess_data('./dataset/data_test.xlsx')
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42)

    # Scale features
    train_features, val_features = scale_features(train_features, val_features)

    # Define and train model
    model = define_model()
    grid_result = hyperparameter_tuning(model, train_features, train_labels)

    # Evaluate the model
    evaluate_model(grid_result, val_features, val_labels)


if __name__ == "__main__":
    main()
