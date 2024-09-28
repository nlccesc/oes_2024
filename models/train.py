import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from metrics import evaluate_performance
import pandas as pd

from rfe import FeatureSelector
from hyperparam_tuning import FTBOptimizer
from stack import ModelStacker
from sklearn.ensemble import RandomForestClassifier


# Load the data from CSV files
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare the data for model training
def prepare_data(df):
    X = df[['altitude_diff', 'lat', 'lon', 
            'accel_forward_change', 'accel_braking_change', 
            'accel_angular_change', 'accel_vertical_change']].values
    
    # Map the operation_kind_id to continuous labels
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
    df['operation_kind_id'] = df['operation_kind_id'].map(label_mapping)
    
    Y = df['operation_kind_id'].values
    return X, Y

# Main function
def main():
    merged_data_path = r'C:\Users\haoha\OneDrive\Desktop\oes_challenge_2024\data\cleaned\engineered_features.csv'
    model_save_dir = r'C:\Users\haoha\OneDrive\Desktop\oes_challenge_2024\models\saved_models'

    # Load and preprocess the data
    merged_data = load_data(merged_data_path)
    X, Y = prepare_data(merged_data)

    # Check for unique classes
    unique_classes = np.unique(Y)
    print(f"Unique classes in the target variable: {unique_classes}")

    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=0.2, random_state=42, 
                                                        stratify=Y)

    # Feature selection using EnRFE
    base_model = RandomForestClassifier(random_state=42)
    feature_selector = FeatureSelector(base_model)
    X_train_selected, selected_features = feature_selector.select_features(X_train, Y_train, 
                                                                           num_features=5)
    X_test_selected = X_test[:, selected_features]

    # Hyperparameter tuning using Freeze-Thaw Bayesian Optimization for RandomForest
    config_space = {
        'n_estimators': range(50, 301),
        'max_depth': range(1, 51),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    }
    
    optimizer = FTBOptimizer(
        model=RandomForestClassifier(random_state=42),
        X_train=X_train_selected,
        y_train=Y_train,
        config_space=config_space,
        max_steps=50,
        budget=100
    )
    
    best_rf_params = optimizer.tune()
    tuned_rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
    tuned_rf_model.fit(X_train_selected, Y_train)

    # Stacking ensemble with tuned RandomForest as the meta learner and other models as base learners
    stacker = ModelStacker()
    stacking_model = stacker.build_stack()
    stacking_model.fit(X_train_selected, Y_train)

    # Predict using the stacked model
    ensemble_predictions = stacking_model.predict(X_test_selected)

    # Evaluate the stacked model using only F1 Score
    f1 = evaluate_performance(Y_test, ensemble_predictions)
    
    print(f"F1 Score: {f1}")

    # Save the models for future use
    os.makedirs(model_save_dir, exist_ok=True)
    joblib.dump(stacking_model, os.path.join(model_save_dir, 'stacking_model.pkl'))

if __name__ == "__main__":
    main()
