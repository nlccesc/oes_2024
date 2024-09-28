# src/loader.py

import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_telemetry_data(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path)
        return df

    def load_operation_labels(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path)
        return df
