import os
from loader import DataLoader
from preprocess import DataPreprocessing
from feature_engineering import FeatureEngineering

raw_data_dir = r'C:\Users\haoha\OneDrive\Desktop\oes_challenge_2024\data\raw'
cleaned_data_dir = r'C:\Users\haoha\OneDrive\Desktop\oes_challenge_2024\data\cleaned'

data_loader = DataLoader(raw_data_dir)

telemetry_data = data_loader.load_telemetry_data('telemetry_for_operations_training.csv')
operation_labels = data_loader.load_operation_labels('operations_labels_training.csv')

data_preprocessing = DataPreprocessing()

cleaned_telemetry = data_preprocessing.preprocess_telemetry(telemetry_data)

merged_data = data_preprocessing.merge_with_labels(cleaned_telemetry, operation_labels)

feature_engineering = FeatureEngineering()
features = feature_engineering.create_features(merged_data)

os.makedirs(cleaned_data_dir, exist_ok=True)
features.to_csv(os.path.join(cleaned_data_dir, 'engineered_features.csv'), index=False)

print(f"Engineered features saved to {os.path.join(cleaned_data_dir, 'engineered_features.csv')}")
